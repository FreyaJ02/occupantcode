import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras import Model
from keras.layers import Input, Dense, Dropout, concatenate
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, Callback

import tensorflow as tf
from objective_rating_metrics.rating import ISO18571

from isoloss_new import ISO18571Loss_DTW


# 设置随机种子以确保可复现性
np.random.seed(42)
tf.random.set_seed(42)


def define_hybrid_model(input_length=100, scalar_length=3, output_length=100):
    """
    定义混合模型：
    LSTM处理CrashPulse时序数据；
    Dense处理Airbag、Gender、Seatbelt标量参数；
    输出HeadAcceleration曲线。

    注意：
    本模型输出的是归一化后的HeadAcceleration。
    后续预测结果需要通过scaler_head进行反归一化。
    """

    input_lstm = Input(shape=(input_length, 1), name='lstm_input')
    input_dense = Input(shape=(scalar_length,), name='dense_input')

    # LSTM分支
    lstm_branch = Bidirectional(LSTM(units=32, return_sequences=True))(input_lstm)
    lstm_branch = Dropout(0.2)(lstm_branch)
    lstm_branch = Bidirectional(LSTM(units=16))(lstm_branch)
    lstm_branch = Dropout(0.2)(lstm_branch)

    # Dense分支
    dense_branch = Dense(16, activation='relu')(input_dense)
    dense_branch = Dropout(0.2)(dense_branch)

    # 融合
    merged = concatenate([lstm_branch, dense_branch])

    # 共享全连接层
    shared = Dense(64, activation='relu')(merged)
    shared = Dropout(0.2)(shared)
    shared = Dense(32, activation='relu')(shared)
    shared = Dropout(0.2)(shared)

    # 输出层：预测归一化后的HeadAcceleration
    head_output = Dense(output_length, name='head_output')(shared)

    model = Model(
        inputs=[input_lstm, input_dense],
        outputs=head_output
    )

    return model


def load_filtered_npy_data(data_dir):
    """
    从npy文件加载所有数据。
    返回原始值，后续在main函数中进行归一化。
    """

    airbag = np.load(os.path.join(data_dir, 'Airbag.npy'))
    gender = np.load(os.path.join(data_dir, 'Gender.npy'))
    seatbelt = np.load(os.path.join(data_dir, 'Seatbelt.npy'))

    crash_pulse = np.load(os.path.join(data_dir, 'CrashPulse.npy'))
    head_acc = np.load(os.path.join(data_dir, 'HeadAcceleration.npy'))

    assert airbag.shape[0] == gender.shape[0] == seatbelt.shape[0] == \
           crash_pulse.shape[0] == head_acc.shape[0], \
           "所有数据文件的第一维度必须一致"

    scalar_data = np.column_stack([airbag, gender, seatbelt])

    return scalar_data, crash_pulse, head_acc


def cosine_annealing_scheduler(total_epochs, lr_max=1e-3, lr_min=1e-6):
    """
    余弦退火学习率调度函数。
    """

    def scheduler(epoch, lr):
        if total_epochs <= 1:
            return lr_min

        cosine_decay = 0.5 * (1 + np.cos(np.pi * epoch / (total_epochs - 1)))
        return lr_min + (lr_max - lr_min) * cosine_decay

    return scheduler


class EpochTimingCallback(Callback):
    """
    记录每个epoch耗时和阶段总耗时。
    """

    def __init__(self, stage_name):
        super().__init__()
        self.stage_name = stage_name
        self.epoch_times = []
        self.total_train_time = 0.0
        self.train_start_time = None
        self.epoch_start_time = None

    def on_train_begin(self, logs=None):
        self.train_start_time = time.perf_counter()
        self.epoch_times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.perf_counter() - self.epoch_start_time
        self.epoch_times.append(elapsed)
        print(f"{self.stage_name} - Epoch {epoch + 1} 用时: {elapsed:.2f} 秒")

    def on_train_end(self, logs=None):
        self.total_train_time = time.perf_counter() - self.train_start_time
        print(f"{self.stage_name} 训练总用时: {self.total_train_time:.2f} 秒")


def train_model_two_stage(model,
                          X_train_lstm, X_train_dense, y_train_head,
                          X_val_lstm, X_val_dense, y_val_head,
                          save_dir):
    """
    两阶段训练：

    第一阶段：
        使用MSE损失函数进行预训练。
        目标是让模型先学习曲线的整体数值范围和基本趋势。

    第二阶段：
        加载第一阶段验证集loss最优权重。
        使用ISO18571Loss_DTW继续微调。
        目标是优化曲线形态和ISO相关评分。

    注意：
        两个阶段均使用归一化后的y_train_head和y_val_head。
    """

    os.makedirs(save_dir, exist_ok=True)

    # =========================================================
    # 第一阶段：MSE预训练
    # =========================================================
    stage1_epochs = 100
    stage1_weights_path = os.path.join(save_dir, 'stage1_mse_best.weights.h5')

    print("\n" + "=" * 60)
    print("第一阶段：使用 MSE 进行预训练")
    print("=" * 60)

    model.compile(
        loss='mse',
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3)
    )

    stage1_lr_scheduler = LearningRateScheduler(
        cosine_annealing_scheduler(
            total_epochs=stage1_epochs,
            lr_max=1e-3,
            lr_min=1e-5
        ),
        verbose=1
    )

    stage1_checkpoint = ModelCheckpoint(
        filepath=stage1_weights_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )

    stage1_timing = EpochTimingCallback(stage_name='Stage 1 MSE')

    history_stage1 = model.fit(
        [X_train_lstm, X_train_dense],
        y_train_head,
        epochs=stage1_epochs,
        batch_size=16,
        validation_data=([X_val_lstm, X_val_dense], y_val_head),
        callbacks=[
            stage1_lr_scheduler,
            stage1_checkpoint,
            stage1_timing
        ],
        verbose=1
    )

    print(f"\n第一阶段最优权重已保存至: {stage1_weights_path}")

    # 加载第一阶段最优权重
    if os.path.exists(stage1_weights_path):
        model.load_weights(stage1_weights_path)
        print("已加载第一阶段验证集loss最优权重，准备进入第二阶段。")
    else:
        print("警告：未找到第一阶段最优权重，将使用当前模型参数进入第二阶段。")

    # =========================================================
    # 第二阶段：ISO18571Loss_DTW微调
    # =========================================================
    stage2_epochs = 100
    stage2_weights_path = os.path.join(save_dir, 'stage2_isoloss_best.weights.h5')

    print("\n" + "=" * 60)
    print("第二阶段：使用 ISO18571Loss_DTW 进行微调")
    print("=" * 60)

    model.compile(
        loss=ISO18571Loss_DTW(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4)
    )

    stage2_lr_scheduler = LearningRateScheduler(
        cosine_annealing_scheduler(
            total_epochs=stage2_epochs,
            lr_max=1e-4,
            lr_min=1e-6
        ),
        verbose=1
    )

    stage2_checkpoint = ModelCheckpoint(
        filepath=stage2_weights_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )

    stage2_timing = EpochTimingCallback(stage_name='Stage 2 ISOLOSS')

    history_stage2 = model.fit(
        [X_train_lstm, X_train_dense],
        y_train_head,
        epochs=stage2_epochs,
        batch_size=16,
        validation_data=([X_val_lstm, X_val_dense], y_val_head),
        callbacks=[
            stage2_lr_scheduler,
            stage2_checkpoint,
            stage2_timing
        ],
        verbose=1
    )

    print(f"\n第二阶段最优权重已保存至: {stage2_weights_path}")

    # 加载第二阶段最优权重，保证最终测试使用的是ISO微调阶段验证集最优模型
    if os.path.exists(stage2_weights_path):
        model.load_weights(stage2_weights_path)
        print("已加载第二阶段验证集loss最优权重，将用于最终测试。")
    else:
        print("警告：未找到第二阶段最优权重，将使用当前模型参数进行测试。")

    # =========================================================
    # 保存两阶段loss日志
    # =========================================================
    loss_rows = []

    stage1_loss = history_stage1.history.get('loss', [])
    stage1_val_loss = history_stage1.history.get('val_loss', [])

    for i in range(len(stage1_loss)):
        loss_rows.append({
            'stage': 'stage1_mse',
            'epoch': i + 1,
            'global_epoch': i + 1,
            'loss': stage1_loss[i],
            'val_loss': stage1_val_loss[i] if i < len(stage1_val_loss) else np.nan,
            'epoch_time_seconds': stage1_timing.epoch_times[i] if i < len(stage1_timing.epoch_times) else np.nan
        })

    offset = len(stage1_loss)

    stage2_loss = history_stage2.history.get('loss', [])
    stage2_val_loss = history_stage2.history.get('val_loss', [])

    for i in range(len(stage2_loss)):
        loss_rows.append({
            'stage': 'stage2_isoloss',
            'epoch': i + 1,
            'global_epoch': offset + i + 1,
            'loss': stage2_loss[i],
            'val_loss': stage2_val_loss[i] if i < len(stage2_val_loss) else np.nan,
            'epoch_time_seconds': stage2_timing.epoch_times[i] if i < len(stage2_timing.epoch_times) else np.nan
        })

    loss_log_df = pd.DataFrame(loss_rows)
    loss_log_path = os.path.join(save_dir, 'two_stage_epoch_loss_history.csv')
    loss_log_df.to_csv(loss_log_path, index=False, encoding='utf-8')
    print(f"两阶段loss记录已保存至: {loss_log_path}")

    # =========================================================
    # 保存训练时间统计
    # =========================================================
    timing_report_path = os.path.join(save_dir, 'two_stage_training_time_summary.txt')

    total_time = stage1_timing.total_train_time + stage2_timing.total_train_time

    with open(timing_report_path, 'w', encoding='utf-8') as f:
        f.write("两阶段训练时间统计\n")
        f.write("=" * 40 + "\n\n")

        f.write("第一阶段：MSE预训练\n")
        f.write("-" * 40 + "\n")
        for idx, t_epoch in enumerate(stage1_timing.epoch_times, start=1):
            f.write(f"Stage 1 - Epoch {idx}: {t_epoch:.4f} 秒\n")
        f.write(f"第一阶段总时间: {stage1_timing.total_train_time:.4f} 秒\n")
        if stage1_timing.epoch_times:
            f.write(f"第一阶段平均每Epoch时间: {np.mean(stage1_timing.epoch_times):.4f} 秒\n")

        f.write("\n第二阶段：ISO18571Loss_DTW微调\n")
        f.write("-" * 40 + "\n")
        for idx, t_epoch in enumerate(stage2_timing.epoch_times, start=1):
            f.write(f"Stage 2 - Epoch {idx}: {t_epoch:.4f} 秒\n")
        f.write(f"第二阶段总时间: {stage2_timing.total_train_time:.4f} 秒\n")
        if stage2_timing.epoch_times:
            f.write(f"第二阶段平均每Epoch时间: {np.mean(stage2_timing.epoch_times):.4f} 秒\n")

        f.write("\n总体统计\n")
        f.write("-" * 40 + "\n")
        f.write(f"两阶段总训练时间: {total_time:.4f} 秒\n")

    print(f"两阶段训练时间统计已保存至: {timing_report_path}")

    return model, history_stage1, history_stage2, stage1_weights_path, stage2_weights_path


def set_log_scale_if_positive(values):
    """
    如果loss全部为正，则使用log坐标。
    避免loss中出现0或负数时画图报错。
    """

    valid_values = []

    for v in values:
        try:
            if np.isfinite(v):
                valid_values.append(v)
        except Exception:
            pass

    if valid_values and np.min(valid_values) > 0:
        plt.yscale('log')


def plot_two_stage_training_history(history_stage1, history_stage2, save_dir):
    """
    绘制两阶段训练loss曲线。

    注意：
    第一阶段loss是MSE；
    第二阶段loss是ISO18571Loss_DTW；
    两者数值含义不同，因此建议分开看。
    """

    os.makedirs(save_dir, exist_ok=True)

    stage1_loss = history_stage1.history.get('loss', [])
    stage1_val_loss = history_stage1.history.get('val_loss', [])

    stage2_loss = history_stage2.history.get('loss', [])
    stage2_val_loss = history_stage2.history.get('val_loss', [])

    # 第一阶段MSE loss
    plt.figure(figsize=(10, 5))
    plt.plot(stage1_loss, label='Stage 1 Training MSE Loss', linewidth=2)
    plt.plot(stage1_val_loss, label='Stage 1 Validation MSE Loss', linewidth=2)
    plt.title('Stage 1 - MSE Pretraining Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    set_log_scale_if_positive(list(stage1_loss) + list(stage1_val_loss))

    stage1_plot_path = os.path.join(save_dir, 'stage1_mse_loss.png')
    plt.savefig(stage1_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"第一阶段MSE损失曲线已保存至: {stage1_plot_path}")

    # 第二阶段ISO loss
    plt.figure(figsize=(10, 5))
    plt.plot(stage2_loss, label='Stage 2 Training ISO Loss', linewidth=2)
    plt.plot(stage2_val_loss, label='Stage 2 Validation ISO Loss', linewidth=2)
    plt.title('Stage 2 - ISO18571Loss_DTW Fine-tuning Loss')
    plt.xlabel('Epoch')
    plt.ylabel('ISO Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    set_log_scale_if_positive(list(stage2_loss) + list(stage2_val_loss))

    stage2_plot_path = os.path.join(save_dir, 'stage2_isoloss_loss.png')
    plt.savefig(stage2_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"第二阶段ISO损失曲线已保存至: {stage2_plot_path}")

    # 两阶段整体曲线
    global_train_loss = list(stage1_loss) + list(stage2_loss)
    global_val_loss = list(stage1_val_loss) + list(stage2_val_loss)

    plt.figure(figsize=(12, 5))
    plt.plot(global_train_loss, label='Training Loss', linewidth=2)
    plt.plot(global_val_loss, label='Validation Loss', linewidth=2)

    split_epoch = len(stage1_loss)
    if split_epoch > 0:
        plt.axvline(
            x=split_epoch - 1,
            linestyle='--',
            linewidth=2,
            label='Switch from MSE to ISO Loss'
        )

    plt.title('Two-stage Training Loss')
    plt.xlabel('Global Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    set_log_scale_if_positive(global_train_loss + global_val_loss)

    global_plot_path = os.path.join(save_dir, 'two_stage_training_loss.png')
    plt.savefig(global_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"两阶段总损失曲线已保存至: {global_plot_path}")

    return stage1_plot_path, stage2_plot_path, global_plot_path


def plot_iso_scores_histograms(detailed_metrics, save_dir):
    """
    绘制ISO评分分布直方图。
    """

    os.makedirs(save_dir, exist_ok=True)

    def filter_valid_scores(scores):
        return [x for x in scores if x is not None]

    plt.figure(figsize=(18, 8))

    metric_keys = ['corridor', 'phase', 'magnitude', 'slope', 'overall']
    colors = ['skyblue', 'lightgreen', 'salmon', 'gold', 'orchid']

    for idx, key in enumerate(metric_keys):
        plt.subplot(2, 3, idx + 1)
        scores = filter_valid_scores(detailed_metrics[key])

        if scores:
            plt.hist(
                scores,
                bins=20,
                color=colors[idx],
                edgecolor='black',
                alpha=0.7
            )

        plt.title(f'{key.capitalize()} Score')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()

    iso_histogram_path = os.path.join(save_dir, 'iso_scores_histograms.png')
    plt.savefig(iso_histogram_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"ISO评分分布直方图已保存至: {iso_histogram_path}")
    return iso_histogram_path


def calculate_iso_scores_for_sample(y_true, y_pred, time_axis, sample_idx, output_name, f):
    """
    计算单个样本的ISO评分。

    注意：
    y_true 和 y_pred 必须是原始物理量，不是归一化值。
    """

    def print_and_write(*args, **kwargs):
        message = ' '.join(str(arg) for arg in args)
        print(*args, **kwargs)
        f.write(message + '\n')
        f.flush()

    true_curve = np.vstack((time_axis, y_true[sample_idx])).T
    pred_curve = np.vstack((time_axis, y_pred[sample_idx])).T

    try:
        iso_rating = ISO18571(
            reference_curve=true_curve,
            comparison_curve=pred_curve
        )

        score = iso_rating.overall_rating()

        corridor = iso_rating.corridor_rating(ndigits=-1)
        phase = iso_rating.phase_rating(ndigits=-1)
        magnitude = iso_rating.magnitude_rating(ndigits=-1)
        slope = iso_rating.slope_rating(ndigits=-1)

        print_and_write(
            f"{output_name} - Sample {sample_idx + 1} - "
            f"ISO Score: {score:.2f} "
            f"(Corridor: {corridor:.2f}, "
            f"Phase: {phase:.2f}, "
            f"Magnitude: {magnitude:.2f}, "
            f"Slope: {slope:.2f})"
        )

        return score, corridor, phase, magnitude, slope

    except Exception as e:
        print_and_write(
            f"{output_name} - Sample {sample_idx + 1} - "
            f"Error calculating ISO score: {e}"
        )
        return None, None, None, None, None


def main():
    # 数据目录
    data_dir = '/public/home/acnebvshuq/js/occupant/dataset/final_dataset_4547'

    # 结果保存目录
    save_dir = '/public/home/acnebvshuq/js/occupant/norm/result/mix'
    os.makedirs(save_dir, exist_ok=True)

    # =========================================================
    # 加载数据
    # =========================================================
    print("加载数据...")
    scalar_data, crash_pulse, head_acc = load_filtered_npy_data(data_dir)

    print(f"标量数据形状: {scalar_data.shape}")
    print(f"CrashPulse形状: {crash_pulse.shape}")
    print(f"HeadAcceleration形状: {head_acc.shape}")

    total_samples = scalar_data.shape[0]

    assert scalar_data.shape[0] == crash_pulse.shape[0] == head_acc.shape[0], \
        "所有数据的样本数量必须一致"

    # =========================================================
    # 数据归一化
    # =========================================================
    scaler_crash = MinMaxScaler()
    scaler_head = MinMaxScaler()

    # 标量数据不归一化
    X_scalar = scalar_data

    # CrashPulse归一化
    crash_pulse_reshaped = crash_pulse.reshape(-1, 1)
    X_crash = scaler_crash.fit_transform(crash_pulse_reshaped).reshape(crash_pulse.shape)
    X_crash = X_crash.reshape((X_crash.shape[0], X_crash.shape[1], 1))

    # HeadAcceleration归一化
    head_acc_reshaped = head_acc.reshape(-1, 1)
    y_head_normalized = scaler_head.fit_transform(head_acc_reshaped).reshape(head_acc.shape)

    # =========================================================
    # 划分训练集、验证集、测试集
    # =========================================================
    indices = np.random.permutation(total_samples)

    train_end = int(0.8 * total_samples)
    val_end = int(0.9 * total_samples)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    X_train_crash = X_crash[train_indices]
    X_train_scalar = X_scalar[train_indices]
    y_train_head = y_head_normalized[train_indices]

    X_val_crash = X_crash[val_indices]
    X_val_scalar = X_scalar[val_indices]
    y_val_head = y_head_normalized[val_indices]

    X_test_crash = X_crash[test_indices]
    X_test_scalar = X_scalar[test_indices]
    y_test_head_normalized = y_head_normalized[test_indices]
    y_test_head_original = head_acc[test_indices]

    print("\n数据集划分完成:")
    print(f"训练集样本数: {len(train_indices)}")
    print(f"验证集样本数: {len(val_indices)}")
    print(f"测试集样本数: {len(test_indices)}")

    # =========================================================
    # 构建模型
    # =========================================================
    model = define_hybrid_model(
        input_length=100,
        scalar_length=3,
        output_length=100
    )

    model.summary()

    # =========================================================
    # 两阶段训练
    # =========================================================
    print("\n开始两阶段训练模型...")

    model, history_stage1, history_stage2, stage1_weights_path, stage2_weights_path = train_model_two_stage(
        model,
        X_train_crash,
        X_train_scalar,
        y_train_head,
        X_val_crash,
        X_val_scalar,
        y_val_head,
        save_dir
    )

    # 绘制两阶段训练曲线
    plot_two_stage_training_history(
        history_stage1,
        history_stage2,
        save_dir
    )

    # =========================================================
    # 测试集评估
    # =========================================================
    print("\n开始在测试集上评估第二阶段最优模型...")

    # 此时model已经加载了第二阶段最佳权重
    model.compile(
        loss=ISO18571Loss_DTW(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4)
    )

    result = model.evaluate(
        [X_test_crash, X_test_scalar],
        y_test_head_normalized,
        verbose=1
    )

    print(f"Test Loss: {result}")

    # =========================================================
    # 预测并反归一化
    # =========================================================
    y_pred_head_normalized = model.predict(
        [X_test_crash, X_test_scalar],
        verbose=1
    )

    y_pred_head_flat = y_pred_head_normalized.reshape(-1, 1)
    y_test_head_norm_flat = y_test_head_normalized.reshape(-1, 1)

    y_pred_head_original = scaler_head.inverse_transform(
        y_pred_head_flat
    ).reshape(y_pred_head_normalized.shape)

    y_test_head_from_norm = scaler_head.inverse_transform(
        y_test_head_norm_flat
    ).reshape(y_test_head_normalized.shape)

    y_test_head_true = y_test_head_original
    y_test_head_pred = y_pred_head_original

    # =========================================================
    # 计算MSE和MAE
    # =========================================================
    head_mse = mean_squared_error(
        y_test_head_true.flatten(),
        y_test_head_pred.flatten()
    )

    head_mae = np.mean(
        np.abs(y_test_head_true - y_test_head_pred)
    )

    # =========================================================
    # 保存测试结果和ISO评分
    # =========================================================
    result_file_path = os.path.join(save_dir, 'result.txt')

    with open(result_file_path, 'w', encoding='utf-8') as f:

        def print_and_write(*args, **kwargs):
            message = ' '.join(str(arg) for arg in args)
            print(*args, **kwargs)
            f.write(message + '\n')
            f.flush()

        print_and_write("\n两阶段训练设置:")
        print_and_write("第一阶段损失函数: MSE")
        print_and_write("第二阶段损失函数: ISO18571Loss_DTW")
        print_and_write("两个阶段均使用归一化后的HeadAcceleration进行训练")
        print_and_write(f"第一阶段最优权重: {stage1_weights_path}")
        print_and_write(f"第二阶段最优权重: {stage2_weights_path}")

        print_and_write("\n测试集评估指标:")
        print_and_write(f"Test ISO Loss: {result}")
        print_and_write(f"HeadAcceleration MSE: {head_mse}")
        print_and_write(f"HeadAcceleration MAE: {head_mae}")

        print_and_write("\n计算ISO评分:")

        time_axis = np.linspace(0, 0.2, 100)

        head_iso_scores = []
        head_corridor_scores = []
        head_phase_scores = []
        head_magnitude_scores = []
        head_slope_scores = []

        num_samples_to_calc = len(y_test_head_true)

        for i in range(num_samples_to_calc):
            print_and_write(f"\nSample {i + 1} ISO评分:")

            head_score, head_corridor, head_phase, head_magnitude, head_slope = calculate_iso_scores_for_sample(
                y_test_head_true,
                y_test_head_pred,
                time_axis,
                i,
                "HeadAcceleration",
                f
            )

            head_iso_scores.append(head_score)
            head_corridor_scores.append(head_corridor)
            head_phase_scores.append(head_phase)
            head_magnitude_scores.append(head_magnitude)
            head_slope_scores.append(head_slope)

        print_and_write("\n所有测试集的平均ISO评分:")

        def filter_valid_scores(scores):
            return [s for s in scores if s is not None]

        valid_head_scores = filter_valid_scores(head_iso_scores)

        if valid_head_scores:
            avg_head_iso = np.mean(valid_head_scores)
            avg_head_corridor = np.mean(filter_valid_scores(head_corridor_scores))
            avg_head_phase = np.mean(filter_valid_scores(head_phase_scores))
            avg_head_magnitude = np.mean(filter_valid_scores(head_magnitude_scores))
            avg_head_slope = np.mean(filter_valid_scores(head_slope_scores))

            print_and_write(
                f"HeadAcceleration - 平均ISO评分: {avg_head_iso:.2f} "
                f"(Corridor: {avg_head_corridor:.2f}, "
                f"Phase: {avg_head_phase:.2f}, "
                f"Magnitude: {avg_head_magnitude:.2f}, "
                f"Slope: {avg_head_slope:.2f})"
            )
        else:
            print_and_write("HeadAcceleration - 没有有效的ISO评分")

        detailed_metrics = {
            'corridor': head_corridor_scores,
            'phase': head_phase_scores,
            'magnitude': head_magnitude_scores,
            'slope': head_slope_scores,
            'overall': head_iso_scores
        }

        iso_histogram_path = plot_iso_scores_histograms(
            detailed_metrics,
            save_dir
        )

        print_and_write(f"ISO评分分布直方图已保存至: {iso_histogram_path}")

    print(f"\n结果已保存至: {result_file_path}")

    # =========================================================
    # 可视化预测结果
    # =========================================================
    images_dir = os.path.join(save_dir, 'prediction_images')
    os.makedirs(images_dir, exist_ok=True)

    num_samples_to_plot = min(20, len(y_test_head_true))

    for i in range(num_samples_to_plot):
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        ax.plot(
            y_test_head_true[i],
            'g-',
            label='True',
            linewidth=1,
            alpha=0.8
        )

        ax.plot(
            y_test_head_pred[i],
            'r--',
            label='Predict',
            linewidth=1,
            alpha=0.8
        )

        if i < len(head_iso_scores) and head_iso_scores[i] is not None:
            ax.text(
                0.02,
                0.98,
                f'ISO Score: {head_iso_scores[i]:.2f}',
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(
                    boxstyle='round',
                    facecolor='white',
                    alpha=0.8
                )
            )

        ax.set_title(f'HeadAcceleration - Sample {i + 1}')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Head Acceleration')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        image_path = os.path.join(
            images_dir,
            f'prediction_sample_{i + 1}.png'
        )

        plt.savefig(
            image_path,
            dpi=300,
            bbox_inches='tight'
        )

        plt.close()

        print(f"预测结果图已保存至: {image_path}")

    return model


if __name__ == "__main__":
    model = main()