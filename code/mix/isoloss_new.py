import tensorflow as tf
import numpy as np
from sdtw_div.numba_ops import sdtw, sdtw_grad_C


class ISO18571Loss_DTW(tf.keras.losses.Loss):
    def __init__(
        self,
        k_z=2.0,
        k_p=1.0,
        k_m=1.0,
        eps_m=0.5,
        e_s=2.0,
        init_min=0.8,
        a_0=0.05,
        b_0=0.5,
        w_z=0.4,
        w_p=0.2,
        w_m=0.2,
        w_s=0.2,
        gamma=0.1,
        use_soft_shift=True,
        shift_tau=0.1,
        dtw_window_size=0.1,

        # ===== 平滑评分相关参数 =====
        smooth_beta=25.0,        # softplus 近似 max(x-th, 0) 的平滑强度，越大越接近硬阈值

        corridor_alpha=2.0,      # 走廊项：内走廊之后的基础惩罚强度
        corridor_lambda=8.0,     # 走廊项：超出外走廊后的附加惩罚强度

        magnitude_alpha=1.0,     # 幅值项：全局基础惩罚强度
        magnitude_lambda=6.0,    # 幅值项：超阈值后的附加惩罚强度

        slope_alpha=1.0,         # 斜率项：全局基础惩罚强度
        slope_lambda=6.0,        # 斜率项：超阈值后的附加惩罚强度
        slope_power=1.0,         # 斜率项：基础幂指数

        phase_alpha=2.0,         # 相位项：连续相位评分强度（不再在最大偏移处硬截断）

        # ===== 为兼容旧接口保留，但在新平滑版中不再使用 =====
        corridor_tail_scale=0.2,
        corridor_tail_decay=20.0,
        magnitude_tail_scale=0.2,
        magnitude_tail_decay=20.0,
        slope_tail_scale=0.2,
        slope_tail_decay=20.0,
        **kwargs
    ):
        """
        平滑版 ISO18571 风格损失函数

        核心变化：
        1. 走廊/幅值/斜率项不再使用硬阈值 + tf.where 切换
        2. 统一改为“基础误差惩罚 + 软超差附加惩罚”的连续评分
        3. 相位项默认也改为连续指数评分，避免边界硬截断

        参数说明（新增部分）：
        - smooth_beta: softplus 平滑强度，越大越接近 max(x-th, 0)
        - *_alpha: 全局基础惩罚强度
        - *_lambda: 超阈值后的附加惩罚强度
        - slope_power: 斜率项基础误差的幂指数
        - phase_alpha: 相位评分衰减强度
        """
        super().__init__(**kwargs)

        self.k_z = float(k_z)
        self.k_p = float(k_p)
        self.k_m = float(k_m)

        self.eps_m = float(eps_m)
        self.e_s = float(e_s)
        self.init_min = float(init_min)
        self.a_0 = float(a_0)
        self.b_0 = float(b_0)

        self.gamma = float(gamma)
        self.use_soft_shift = bool(use_soft_shift)
        self.shift_tau = float(shift_tau)
        self.dtw_window_size = float(dtw_window_size)

        # 新平滑参数
        self.smooth_beta = float(smooth_beta)

        self.corridor_alpha = float(corridor_alpha)
        self.corridor_lambda = float(corridor_lambda)

        self.magnitude_alpha = float(magnitude_alpha)
        self.magnitude_lambda = float(magnitude_lambda)

        self.slope_alpha = float(slope_alpha)
        self.slope_lambda = float(slope_lambda)
        self.slope_power = float(slope_power)

        self.phase_alpha = float(phase_alpha)

        # 旧参数保留（兼容旧初始化代码），但平滑版不再使用
        self.corridor_tail_scale = float(corridor_tail_scale)
        self.corridor_tail_decay = float(corridor_tail_decay)
        self.magnitude_tail_scale = float(magnitude_tail_scale)
        self.magnitude_tail_decay = float(magnitude_tail_decay)
        self.slope_tail_scale = float(slope_tail_scale)
        self.slope_tail_decay = float(slope_tail_decay)

        weights_sum = w_z + w_p + w_m + w_s
        if abs(weights_sum - 1.0) >= 1e-6:
            raise ValueError("权重总和必须为 1")

        self.w_z = float(w_z)
        self.w_p = float(w_p)
        self.w_m = float(w_m)
        self.w_s = float(w_s)

        # 仍保留 max_shift 作为相位搜索窗口比例
        self.max_shift = round(1.0 - self.init_min, 2)

    # =========================
    # 主调用
    # =========================

    def call(self, y_true, y_pred):
        """
        参数:
        - y_true: (batch_size, seq_len)
        - y_pred: (batch_size, seq_len)

        返回:
        - 平均损失标量
        """
        batch_size = tf.shape(y_pred)[0]

        def process_single_curve(i):
            pred_curve = tf.cast(y_pred[i], tf.float32)
            true_curve = tf.cast(y_true[i], tf.float32)

            if self.use_soft_shift:
                shifted_pred, shifted_true, e_p = self._soft_shift_and_phase(
                    pred_curve, true_curve
                )
                z = self._corridor_rating(pred_curve, true_curve)
            else:
                shifted_pred, shifted_true = self._get_shifted_curves(
                    pred_curve, true_curve
                )
                z = self._corridor_rating(pred_curve, true_curve)
                e_p = self._phase_rating(pred_curve, true_curve)

            e_m = self._magnitude_rating(shifted_pred, shifted_true)
            e_s = self._slope_rating(shifted_pred, shifted_true)

            overall_rating = (
                self.w_z * z +
                self.w_p * e_p +
                self.w_m * e_m +
                self.w_s * e_s
            )

            return 1.0 - overall_rating

        losses = tf.map_fn(
            process_single_curve,
            tf.range(batch_size),
            fn_output_signature=tf.float32
        )
        return tf.reduce_mean(losses)

    # =========================
    # 平滑辅助函数
    # =========================

    def _soft_excess(self, x, threshold, beta=None):
        """
        平滑近似 max(x - threshold, 0)

        公式:
        soft_excess(x) =
            [softplus(beta*(x-th)) - softplus(-beta*th)] / beta

        这样在 x=0 时近似为 0，并且整体连续可导。
        """
        x = tf.cast(x, tf.float32)
        threshold = tf.cast(threshold, tf.float32)
        beta = tf.cast(self.smooth_beta if beta is None else beta, tf.float32)

        return (
            tf.nn.softplus(beta * (x - threshold)) -
            tf.nn.softplus(-beta * threshold)
        ) / beta

    def _phase_score_from_ratio(self, shift_ratio):
        """
        连续相位评分：不再在 shift_ratio>=1 时硬置 0
        e_p = exp(-phase_alpha * shift_ratio^k_p)
        """
        shift_ratio = tf.maximum(tf.cast(shift_ratio, tf.float32), 0.0)
        return tf.exp(
            -tf.cast(self.phase_alpha, tf.float32) *
            tf.pow(shift_ratio, tf.cast(self.k_p, tf.float32))
        )

    # =========================
    # 相位部分
    # =========================

    def _soft_shift_and_phase(self, pred, true):
        """
        可微软平移 + 连续相位评分
        """
        pred = tf.cast(pred, tf.float32)
        true = tf.cast(true, tf.float32)

        n = tf.shape(true)[0]
        max_shift_points = tf.cast(
            tf.round(tf.cast(n, tf.float32) * tf.cast(self.max_shift, tf.float32)),
            tf.int32
        )
        max_shift_points = tf.maximum(max_shift_points, 1)

        lags = tf.range(-max_shift_points, max_shift_points + 1, dtype=tf.int32)

        def per_lag(lag):
            lag = tf.cast(lag, tf.int32)
            abs_lag = tf.abs(lag)

            def pos_case():
                # lag > 0: pred[lag:] 对 true[:-lag]
                p_seg = pred[lag:]
                t_seg = true[:-lag]
                pad = tf.zeros([lag], dtype=tf.float32)
                mask = tf.concat(
                    [tf.ones([tf.shape(p_seg)[0]], tf.float32),
                     tf.zeros([lag], tf.float32)],
                    axis=0
                )
                return (
                    tf.concat([p_seg, pad], axis=0),
                    tf.concat([t_seg, pad], axis=0),
                    mask,
                    p_seg,
                    t_seg
                )

            def neg_case():
                # lag < 0: pred[:lag] 对 true[-lag:]
                k = -lag
                p_seg = pred[:lag]
                t_seg = true[k:]
                pad = tf.zeros([k], dtype=tf.float32)
                mask = tf.concat(
                    [tf.ones([tf.shape(p_seg)[0]], tf.float32),
                     tf.zeros([k], tf.float32)],
                    axis=0
                )
                return (
                    tf.concat([p_seg, pad], axis=0),
                    tf.concat([t_seg, pad], axis=0),
                    mask,
                    p_seg,
                    t_seg
                )

            def zero_case():
                mask = tf.ones([n], tf.float32)
                return pred, true, mask, pred, true

            pred_pad, true_pad, mask, p_seg, t_seg = tf.case(
                [
                    (tf.greater(lag, 0), pos_case),
                    (tf.less(lag, 0), neg_case),
                ],
                default=zero_case,
                exclusive=True
            )

            corr = tf.reduce_sum(p_seg * t_seg) / (
                tf.norm(p_seg) * tf.norm(t_seg) + 1e-8
            )

            return (
                corr,
                pred_pad,
                true_pad,
                mask,
                tf.cast(abs_lag, tf.float32),
            )

        corrs, preds_pad, trues_pad, masks, abs_lags_f = tf.map_fn(
            per_lag,
            lags,
            fn_output_signature=(
                tf.TensorSpec([], tf.float32),
                tf.TensorSpec([None], tf.float32),
                tf.TensorSpec([None], tf.float32),
                tf.TensorSpec([None], tf.float32),
                tf.TensorSpec([], tf.float32),
            )
        )

        tau = tf.maximum(tf.cast(self.shift_tau, tf.float32), 1e-4)
        w = tf.nn.softmax(corrs / tau, axis=0)
        w2 = tf.reshape(w, [-1, 1])

        num_pred = tf.reduce_sum(w2 * preds_pad, axis=0)
        num_true = tf.reduce_sum(w2 * trues_pad, axis=0)
        den = tf.reduce_sum(w2 * masks, axis=0) + 1e-8

        shifted_pred = num_pred / den
        shifted_true = num_true / den

        expected_abs_lag = tf.reduce_sum(w * abs_lags_f)
        shift_ratio = expected_abs_lag / (tf.cast(max_shift_points, tf.float32) + 1e-8)
        e_p = self._phase_score_from_ratio(shift_ratio)

        shifted_pred.set_shape(pred.shape)
        shifted_true.set_shape(true.shape)
        e_p.set_shape(())

        return shifted_pred, shifted_true, e_p

    def _get_shifted_curves(self, pred, true):
        """
        硬 best-lag 对齐（不可微），仅在 use_soft_shift=False 时使用
        """
        max_shift_points = tf.cast(
            tf.round(tf.cast(tf.shape(true)[0], tf.float32) * self.max_shift),
            tf.int32
        )
        max_shift_points = tf.maximum(max_shift_points, 1)

        def cross_corr(a, b, lag):
            a_len = tf.shape(a)[0]
            b_len = tf.shape(b)[0]
            a_seg = a[tf.maximum(lag, 0):tf.minimum(a_len, a_len + lag)]
            b_seg = b[tf.maximum(-lag, 0):tf.minimum(b_len, b_len - lag)]
            return tf.reduce_sum(a_seg * b_seg) / (
                tf.norm(a_seg) * tf.norm(b_seg) + 1e-8
            )

        lags = tf.range(-max_shift_points, max_shift_points + 1)
        correlations = tf.map_fn(
            lambda lag: cross_corr(pred, true, lag),
            lags,
            fn_output_signature=tf.float32
        )
        best_lag = lags[tf.argmax(correlations)]

        def shift_curve(pred_in, true_in, lag):
            lag = tf.cast(lag, tf.int32)
            return tf.cond(
                lag > 0,
                lambda: (pred_in[lag:], true_in[:-lag]),
                lambda: (pred_in[:lag], true_in[-lag:])
            )

        return tf.cond(
            tf.equal(best_lag, 0),
            lambda: (pred, true),
            lambda: shift_curve(pred, true, best_lag)
        )

    def _phase_rating(self, pred_curve, true_curve):
        """
        硬 best-lag 相位评分，但评分本身改为连续指数形式
        """
        max_shift_points = tf.cast(
            tf.round(tf.cast(tf.shape(true_curve)[0], tf.float32) * self.max_shift),
            tf.int32
        )
        max_shift_points = tf.maximum(max_shift_points, 1)

        def cross_corr(a, b, lag):
            a_len = tf.shape(a)[0]
            b_len = tf.shape(b)[0]
            a_seg = a[tf.maximum(lag, 0):tf.minimum(a_len, a_len + lag)]
            b_seg = b[tf.maximum(-lag, 0):tf.minimum(b_len, b_len - lag)]
            return tf.reduce_sum(a_seg * b_seg) / (
                tf.norm(a_seg) * tf.norm(b_seg) + 1e-8
            )

        lags = tf.range(-max_shift_points, max_shift_points + 1)
        correlations = tf.map_fn(
            lambda lag: cross_corr(pred_curve, true_curve, lag),
            lags,
            fn_output_signature=tf.float32
        )
        best_lag = lags[tf.argmax(correlations)]

        shift_ratio = tf.abs(tf.cast(best_lag, tf.float32)) / (
            tf.cast(max_shift_points, tf.float32) + 1e-8
        )
        return self._phase_score_from_ratio(shift_ratio)

    # =========================
    # 走廊评分（平滑版）
    # =========================

    def _corridor_rating(self, pred_curve, true_curve):
        """
        平滑版走廊评分

        原逻辑：
        - inner 内 = 1
        - inner~outer 之间按幂函数下降
        - outer 外切换成 tail

        新逻辑：
        - after_inner: 平滑描述“超过 inner 的程度”
        - after_outer: 平滑描述“超过 outer 的程度”
        - 统一写成一个指数评分，无断点
        """
        t_norm = tf.reduce_max(tf.abs(true_curve)) + 1e-8
        inner_corridor = self.a_0 * t_norm
        outer_corridor = self.b_0 * t_norm

        abs_diff = tf.abs(pred_curve - true_curve)

        after_inner = self._soft_excess(abs_diff, inner_corridor) / (
            outer_corridor - inner_corridor + 1e-8
        )
        after_outer = self._soft_excess(abs_diff, outer_corridor) / (
            outer_corridor + 1e-8
        )

        c_i = tf.exp(
            -tf.cast(self.corridor_alpha, tf.float32) *
            tf.pow(after_inner, tf.cast(self.k_z, tf.float32))
            -tf.cast(self.corridor_lambda, tf.float32) *
            after_outer
        )

        return tf.reduce_mean(c_i)

    # =========================
    # 幅值评分（平滑版）
    # =========================

    def _magnitude_rating(self, pred_curve, true_curve):
        """
        平滑版幅值评分

        rel_error = ||aligned_pred - aligned_true||_1 / ||true||_1

        新评分：
        e_m = exp(
            - magnitude_alpha * (rel_error / eps_m)^k_m
            - magnitude_lambda * soft_excess(rel_error, eps_m) / eps_m
        )
        """
        aligned_pred, aligned_true = self._align_curves_dtw(pred_curve, true_curve)

        rel_error = tf.norm(aligned_pred - aligned_true, ord=1) / (
            tf.norm(true_curve, ord=1) + 1e-8
        )

        base = tf.pow(
            rel_error / (tf.cast(self.eps_m, tf.float32) + 1e-8),
            tf.cast(self.k_m, tf.float32)
        )
        excess = self._soft_excess(rel_error, self.eps_m) / (
            tf.cast(self.eps_m, tf.float32) + 1e-8
        )

        e_m = tf.exp(
            -tf.cast(self.magnitude_alpha, tf.float32) * base
            -tf.cast(self.magnitude_lambda, tf.float32) * excess
        )
        return e_m

    # =========================
    # Soft-DTW 对齐
    # =========================

    def _align_curves_dtw(self, pred, true):
        """
        Soft-DTW 曲线对齐（可反传封装）
        """
        gamma = float(self.gamma)
        window_size = float(self.dtw_window_size)

        @tf.custom_gradient
        def align_op(pred_in, true_in):
            def numpy_expected_alignment(pred_np, true_np):
                x = pred_np.reshape(-1, 1)
                y = true_np.reshape(-1, 1)
                _, p = sdtw(x, y, gamma=gamma, window_size=window_size, return_all=True)
                e = sdtw_grad_C(p).astype(np.float32)
                return e

            e = tf.numpy_function(
                numpy_expected_alignment,
                [pred_in, true_in],
                tf.float32
            )
            e.set_shape([None, None])

            aligned_pred = tf.linalg.matvec(e, true_in)
            aligned_true = tf.linalg.matvec(tf.transpose(e), pred_in)

            def grad(d_aligned_pred, d_aligned_true):
                # 一阶近似：忽略 E 对输入的高阶依赖
                d_pred = tf.linalg.matvec(e, d_aligned_true)
                d_true = tf.linalg.matvec(tf.transpose(e), d_aligned_pred)
                return d_pred, d_true

            return (aligned_pred, aligned_true), grad

        pred = tf.cast(pred, tf.float32)
        true = tf.cast(true, tf.float32)

        aligned_pred, aligned_true = align_op(pred, true)
        aligned_pred.set_shape(pred.shape)
        aligned_true.set_shape(true.shape)
        return aligned_pred, aligned_true

    # =========================
    # 斜率评分（平滑版）
    # =========================

    def _slope_rating(self, pred_curve, true_curve):
        """
        平滑版斜率评分
        """
        def compute_gradient(y):
            y_pad = tf.pad(y, paddings=[[1, 1]], mode="SYMMETRIC")
            return (y_pad[2:] - y_pad[:-2]) / 0.0002

        pred_grad = compute_gradient(pred_curve)
        true_grad = compute_gradient(true_curve)

        pred_smooth = tf.zeros_like(pred_grad)
        true_smooth = tf.zeros_like(true_grad)

        def apply_smoothing(grad, smooth_arr):
            # 前4点和后4点：窗口 1,3,5,7
            for idx, window in enumerate([1, 3, 5, 7]):
                smooth_arr = tf.tensor_scatter_nd_update(
                    smooth_arr,
                    [[idx]],
                    [tf.reduce_mean(grad[:window])]
                )

                end_idx = tf.shape(smooth_arr, out_type=tf.int32)[0] - 1 - idx
                smooth_arr = tf.tensor_scatter_nd_update(
                    smooth_arr,
                    tf.reshape(end_idx, [1, 1]),
                    [tf.reduce_mean(grad[-window:])]
                )

            # 中间区域：窗口 9
            kernel = tf.ones(9, dtype=tf.float32) / 9.0
            center_conv = tf.nn.conv1d(
                tf.reshape(grad, [1, -1, 1]),
                tf.reshape(kernel, [-1, 1, 1]),
                stride=1,
                padding="VALID"
            )

            return tf.concat(
                [
                    smooth_arr[:4],
                    tf.reshape(center_conv, [-1]),
                    smooth_arr[-4:]
                ],
                axis=0
            )

        pred_smooth = apply_smoothing(pred_grad, pred_smooth)
        true_smooth = apply_smoothing(true_grad, true_smooth)

        rel_error = tf.norm(pred_smooth - true_smooth, ord=1) / (
            tf.norm(true_smooth, ord=1) + 1e-8
        )

        base = tf.pow(
            rel_error / (tf.cast(self.e_s, tf.float32) + 1e-8),
            tf.cast(self.slope_power, tf.float32)
        )
        excess = self._soft_excess(rel_error, self.e_s) / (
            tf.cast(self.e_s, tf.float32) + 1e-8
        )

        e_s = tf.exp(
            -tf.cast(self.slope_alpha, tf.float32) * base
            -tf.cast(self.slope_lambda, tf.float32) * excess
        )
        return e_s