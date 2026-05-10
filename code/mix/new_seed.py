#归一化处理参考PPT
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras import Model
from keras.layers import Input, Dense, Dropout, concatenate
from keras.layers import LSTM
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.models import load_model
from keras.layers import Bidirectional
import tensorflow as tf
from keras import backend as K
import sys
sys.path.append(r'/public/home/acnebvshuq/js/occupant/ISO18571-main')
from objective_rating_metrics.rating import ISO18571
from sklearn.model_selection import KFold
import time
import pickle
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, Callback, EarlyStopping
import random


def define_hybrid_model(input_length=100, scalar_length=3, output_length=100):
    input_lstm = Input(shape=(input_length, 1), name='lstm_input')
    input_dense = Input(shape=(scalar_length,), name='dense_input')

    lstm_branch = Bidirectional(LSTM(units=128, return_sequences=True))(input_lstm)
    lstm_branch = Dropout(0.2)(lstm_branch)
    lstm_branch = Bidirectional(LSTM(units=64))(lstm_branch)
    lstm_branch = Dropout(0.2)(lstm_branch)

    dense_branch = Dense(16, activation='relu')(input_dense)
    dense_branch = Dropout(0.2)(dense_branch)

    merged = concatenate([lstm_branch, dense_branch])

    shared = Dense(128, activation='relu')(merged)
    shared = Dropout(0.2)(shared)
    shared = Dense(64, activation='relu')(shared)
    shared = Dropout(0.2)(shared)

    head_output = Dense(output_length, name='head_output')(shared)
    neck_force_output = Dense(output_length, name='neck_force_output')(shared)
    neck_moment_output = Dense(output_length, name='neck_moment_output')(shared)
    chest_output = Dense(output_length, name='chest_output')(shared)

    model = Model(inputs=[input_lstm, input_dense],
                  outputs=[head_output, neck_force_output, neck_moment_output, chest_output])

    model.compile(
        loss={'head_output': 'mse', 'neck_force_output': 'mse', 'neck_moment_output': 'mse', 'chest_output': 'mse'},
        loss_weights={'head_output': 1.0, 'neck_force_output': 1.0, 'neck_moment_output': 1.0, 'chest_output': 1.0},
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3)
    )
    return model
def load_filtered_npy_data(data_dir):
    airbag = np.load(os.path.join(data_dir, 'Airbag.npy'))
    gender = np.load(os.path.join(data_dir, 'Gender.npy'))
    seatbelt = np.load(os.path.join(data_dir, 'Seatbelt.npy'))
    crash_pulse = np.load(os.path.join(data_dir, 'CrashPulse.npy'))
    head_acc = np.load(os.path.join(data_dir, 'HeadAcceleration.npy'))
    neck_force = np.load(os.path.join(data_dir, 'NeckForce.npy'))
    neck_moment = np.load(os.path.join(data_dir, 'NeckMoment.npy'))
    chest_disp = np.load(os.path.join(data_dir, 'ChestDisplacement.npy'))

    assert airbag.shape[0] == gender.shape[0] == seatbelt.shape[0] == crash_pulse.shape[0] == head_acc.shape[0] == neck_force.shape[0] == neck_moment.shape[0] == chest_disp.shape[0], \
        "所有数据文件的第一维度必须一致"

    scalar_data = np.column_stack([airbag, gender, seatbelt])
    return scalar_data, crash_pulse, head_acc, neck_force, neck_moment, chest_disp
def cosine_annealing_scheduler(total_epochs, lr_max=1e-3, lr_min=1e-6, log_dir="./"):
    log_file = os.path.join(log_dir, "lr_change.txt")
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("epoch\tlearning_rate\n")

    def scheduler(epoch, lr):
        if total_epochs <= 1:
            current_lr = lr_min
        else:
            cosine_decay = 0.5 * (1 + np.cos(np.pi * epoch / (total_epochs - 1)))
            current_lr = lr_min + (lr_max - lr_min) * cosine_decay
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"{epoch}\t{current_lr:.8e}\n")
        return current_lr

    return scheduler
def train_model(model, X_train_lstm, X_train_dense, y_train_head, y_train_neck_force, y_train_neck_moment, y_train_chest,
                X_val_lstm, X_val_dense, y_val_head, y_val_neck_force, y_val_neck_moment, y_val_chest, save_dir):
    class EpochTimingCallback(Callback):
        def on_train_begin(self, logs=None):
            self.train_start_time = time.perf_counter()
            self.epoch_times = []
            self.lrs = []  # 记录每个epoch的学习率

        def on_epoch_begin(self, epoch, logs=None):
            self.epoch_start_time = time.perf_counter()
            # 记录当前epoch使用的学习率（调度器已更新）
            lr = tf.keras.backend.get_value(self.model.optimizer.lr)
            self.lrs.append(lr)

        def on_epoch_end(self, epoch, logs=None):
            elapsed = time.perf_counter() - self.epoch_start_time
            self.epoch_times.append(elapsed)
            print(f"Epoch {epoch + 1} 用时: {elapsed:.2f} 秒")

        def on_train_end(self, logs=None):
            self.total_train_time = time.perf_counter() - self.train_start_time
            print(f"训练总用时: {self.total_train_time:.2f} 秒")

    timing_callback = EpochTimingCallback()
    total_epochs = 300  # 增大轮数
    lr_scheduler = LearningRateScheduler(
        cosine_annealing_scheduler(total_epochs, lr_max=1e-3, lr_min=1e-6, log_dir=save_dir), verbose=1
    )
    model_path = os.path.join(save_dir, 'best_opendata_lstm_model.h5')
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=1)
    # early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1)

    history = model.fit(
        [X_train_lstm, X_train_dense],
        [y_train_head, y_train_neck_force, y_train_neck_moment, y_train_chest],
        epochs=total_epochs,
        batch_size=16,
        validation_data=([X_val_lstm, X_val_dense],
                         [y_val_head, y_val_neck_force, y_val_neck_moment, y_val_chest]),
        callbacks=[lr_scheduler, checkpoint, timing_callback],
        verbose=1
    )

    # ========== 新增：保存 epoch 时间与损失详细信息 ==========
    epochs = len(history.history['loss'])
    # 构建 DataFrame
    df_epoch = pd.DataFrame({
        'epoch': range(1, epochs + 1),
        'loss': history.history['loss'],
        'val_loss': history.history['val_loss'],
        'lr': timing_callback.lrs[:epochs],
        'epoch_time': timing_callback.epoch_times[:epochs]
    })
    # 保存 epoch_loss_history.csv
    csv_path = os.path.join(save_dir, 'epoch_loss_history.csv')
    df_epoch.to_csv(csv_path, index=False)
    print(f"各epoch损失与时间已保存至: {csv_path}")

    # 保存 epoch_time.txt（格式：epoch time_seconds，用空格分隔）
    txt_path = os.path.join(save_dir, 'epoch_time.txt')
    with open(txt_path, 'w') as f:
        f.write("epoch time_seconds\n")
        for idx, t in enumerate(timing_callback.epoch_times[:epochs], start=1):
            f.write(f"{idx} {t:.8f}\n")   # 与您示例一致，保留足够小数
    print(f"各epoch时间已保存至: {txt_path}")
    # ========== 原有时间统计汇总继续保留 ==========
    timing_report_path = os.path.join(save_dir, 'training_time_summary.txt')
    with open(timing_report_path, 'w', encoding='utf-8') as f:
        f.write("训练时间统计\n")
        f.write("=" * 30 + "\n")
        for idx, t_epoch in enumerate(timing_callback.epoch_times, start=1):
            f.write(f"Epoch {idx}: {t_epoch:.4f} 秒\n")
        f.write("-" * 30 + "\n")
        f.write(f"总训练时间: {timing_callback.total_train_time:.4f} 秒\n")
        if timing_callback.epoch_times:
            f.write(f"平均每Epoch时间: {np.mean(timing_callback.epoch_times):.4f} 秒\n")
    print(f"训练时间统计已保存至: {timing_report_path}")

    return history
def plot_training_history(history, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    history_dict = history.history
    plt.figure(figsize=(10, 5))
    if 'loss' in history_dict:
        plt.plot(history_dict['loss'], 'k-', label='Total Training Loss', linewidth=2)
    if 'val_loss' in history_dict:
        plt.plot(history_dict['val_loss'], 'k--', label='Total Validation Loss', linewidth=2)
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    total_loss_plot_path = os.path.join(save_dir, 'total_training_loss.png')
    plt.savefig(total_loss_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"总损失曲线图已保存至: {total_loss_plot_path}")
    return total_loss_plot_path
def plot_metrics_histograms(all_detailed_metrics, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    def filter_valid_scores(scores):
        return [x for x in scores if x is not None]

    output_names = ['HeadAcceleration', 'NeckForce', 'NeckMoment', 'ChestDisplacement']
    for output_name, detailed_metrics in zip(output_names, all_detailed_metrics):
        plt.figure(figsize=(18, 8))
        metric_keys = ['corridor', 'phase', 'magnitude', 'slope', 'overall']
        colors = ['skyblue', 'lightgreen', 'salmon', 'gold', 'orchid']
        for idx, key in enumerate(metric_keys):
            plt.subplot(2, 3, idx + 1)
            scores = filter_valid_scores(detailed_metrics[key])
            if scores:
                plt.hist(scores, bins=20, color=colors[idx], edgecolor='black', alpha=0.7)
            plt.title(f'{output_name} - {key.capitalize()} Score')
            plt.xlabel('Score')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
        plt.tight_layout()
        iso_histogram_path = os.path.join(save_dir, f'{output_name}_metrics_histograms.png')
        plt.savefig(iso_histogram_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"{output_name} ISO评分分布直方图已保存至: {iso_histogram_path}")
    return save_dir
def calculate_iso_scores_for_sample(y_true, y_pred, time_axis, sample_idx, output_name, f):
    def print_and_write(*args, **kwargs):
        message = ' '.join(str(arg) for arg in args)
        print(*args, **kwargs)
        f.write(message + '\n')
        f.flush()

    true_curve = np.vstack((time_axis, y_true[sample_idx])).T
    pred_curve = np.vstack((time_axis, y_pred[sample_idx])).T
    try:
        iso_rating = ISO18571(reference_curve=true_curve, comparison_curve=pred_curve)
        score = iso_rating.overall_rating()
        corridor = iso_rating.corridor_rating(ndigits=-1)
        phase = iso_rating.phase_rating(ndigits=-1)
        magnitude = iso_rating.magnitude_rating(ndigits=-1)
        slope = iso_rating.slope_rating(ndigits=-1)
        print_and_write(
            f"{output_name} - Sample {sample_idx + 1} - ISO Score: {score:.2f} (Corridor: {corridor:.2f}, Phase: {phase:.2f}, Magnitude: {magnitude:.2f}, Slope: {slope:.2f})")
        return score, corridor, phase, magnitude, slope
    except Exception as e:
        print_and_write(f"{output_name} - Sample {sample_idx + 1} - Error calculating ISO score: {e}")
        return None, None, None, None, None
def main():
    seed = 33
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    data_dir = r'/public/home/acnebvshuq/js/occupant/dataset/newQ'
    save_dir = fr'/public/home/acnebvshuq/js/occupant/norm/result/seed{seed}'
    os.makedirs(save_dir, exist_ok=True)

    print("加载数据...")
    scalar_data, crash_pulse, head_acc, neck_force, neck_moment, chest_disp = load_filtered_npy_data(data_dir)
    total_samples = scalar_data.shape[0]
    print(f"\n数据加载完成，总样本数: {total_samples}")
    print(f"标量数据形状: {scalar_data.shape}")
    print(f"CrashPulse形状: {crash_pulse.shape}")
    print(f"HeadAcceleration形状: {head_acc.shape}")
    print(f"NeckForce形状: {neck_force.shape}")
    print(f"NeckMoment形状: {neck_moment.shape}")
    print(f"ChestDisplacement形状: {chest_disp.shape}")

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    all_fold_results = []
    # 用于收集所有折的详细评分（每个输出一个字典，每个键对应一个列表）
    all_head_metrics = {'corridor' : [], 'phase' : [], 'magnitude' : [], 'slope' : [], 'overall' : []}
    all_neck_force_metrics = {'corridor' : [], 'phase' : [], 'magnitude' : [], 'slope' : [], 'overall' : []}
    all_neck_moment_metrics = {'corridor' : [], 'phase' : [], 'magnitude' : [], 'slope' : [], 'overall' : []}
    all_chest_metrics = {'corridor' : [], 'phase' : [], 'magnitude' : [], 'slope' : [], 'overall' : []}

    for fold, (train_index, val_index) in enumerate(kf.split(crash_pulse)):
        print(f"第 {fold + 1} 折开始训练")

        # 折内数据划分
        X_train_scalar = scalar_data[train_index]
        X_val_scalar = scalar_data[val_index]

        crash_train = crash_pulse[train_index]
        crash_val = crash_pulse[val_index]

        y_train_head_raw = head_acc[train_index]
        y_val_head_raw = head_acc[val_index]

        y_train_neck_force_raw = neck_force[train_index]
        y_val_neck_force_raw = neck_force[val_index]

        y_train_neck_moment_raw = neck_moment[train_index]
        y_val_neck_moment_raw = neck_moment[val_index]

        y_train_chest_raw = chest_disp[train_index]
        y_val_chest_raw = chest_disp[val_index]

        print(f"训练集样本数: {len(train_index)}, 验证集样本数: {len(val_index)}")

        # 归一化
        scaler_crash = MinMaxScaler()
        scaler_head = MinMaxScaler()
        scaler_neck_force = MinMaxScaler(feature_range=(-1, 1))
        scaler_neck_moment = MinMaxScaler(feature_range=(-1, 1))
        scaler_chest = MinMaxScaler(feature_range=(-1, 0))

        X_train_crash = scaler_crash.fit_transform(crash_train.reshape(-1, 1)).reshape(-1, 100, 1)
        X_val_crash = scaler_crash.transform(crash_val.reshape(-1, 1)).reshape(-1, 100, 1)

        y_train_head = scaler_head.fit_transform(y_train_head_raw.reshape(-1, 1)).reshape(y_train_head_raw.shape)
        y_val_head = scaler_head.transform(y_val_head_raw.reshape(-1, 1)).reshape(y_val_head_raw.shape)

        y_train_neck_force = scaler_neck_force.fit_transform(y_train_neck_force_raw.reshape(-1, 1)).reshape(y_train_neck_force_raw.shape)
        y_val_neck_force = scaler_neck_force.transform(y_val_neck_force_raw.reshape(-1, 1)).reshape(y_val_neck_force_raw.shape)

        y_train_neck_moment = scaler_neck_moment.fit_transform(y_train_neck_moment_raw.reshape(-1, 1)).reshape(y_train_neck_moment_raw.shape)
        y_val_neck_moment = scaler_neck_moment.transform(y_val_neck_moment_raw.reshape(-1, 1)).reshape(y_val_neck_moment_raw.shape)

        y_train_chest = scaler_chest.fit_transform(y_train_chest_raw.reshape(-1, 1)).reshape(y_train_chest_raw.shape)
        y_val_chest = scaler_chest.transform(y_val_chest_raw.reshape(-1, 1)).reshape(y_val_chest_raw.shape)

        # 构建保存目录
        fold_save_dir = os.path.join(save_dir, f'fold_{fold + 1}')
        os.makedirs(fold_save_dir, exist_ok=True)

        model = define_hybrid_model(input_length=100, scalar_length=3, output_length=100)
        if fold == 0:
            model.summary()

        print("开始训练模型...")
        history = train_model(
            model,
            X_train_crash, X_train_scalar,
            y_train_head, y_train_neck_force, y_train_neck_moment, y_train_chest,
            X_val_crash, X_val_scalar,
            y_val_head, y_val_neck_force, y_val_neck_moment, y_val_chest,
            fold_save_dir
        )
        # 绘制损失曲线（保存到当前折目录）
        loss_plot_path = plot_training_history(history, fold_save_dir)

        # 加载最佳模型
        best_model_path = os.path.join(fold_save_dir, 'best_opendata_lstm_model.h5')
        if os.path.exists(best_model_path):
            print(f"\n加载最佳模型: {best_model_path}")
            model = load_model(best_model_path)

        # 预测
        y_pred_head, y_pred_neck_force, y_pred_neck_moment, y_pred_chest = model.predict(
            [X_val_crash, X_val_scalar], verbose=1
        )

        # 反归一化
        y_test_head_true = y_val_head_raw
        y_test_head_pred = scaler_head.inverse_transform(y_pred_head.reshape(-1, 1)).reshape(y_pred_head.shape)

        y_test_neck_force_true = y_val_neck_force_raw
        y_test_neck_force_pred = scaler_neck_force.inverse_transform(y_pred_neck_force.reshape(-1, 1)).reshape(y_pred_neck_force.shape)

        y_test_neck_moment_true = y_val_neck_moment_raw
        y_test_neck_moment_pred = scaler_neck_moment.inverse_transform(y_pred_neck_moment.reshape(-1, 1)).reshape(y_pred_neck_moment.shape)

        y_test_chest_true = y_val_chest_raw
        y_test_chest_pred = scaler_chest.inverse_transform(y_pred_chest.reshape(-1, 1)).reshape(y_pred_chest.shape)

        # 计算 MSE / MAE
        head_mse = mean_squared_error(y_test_head_true.flatten(), y_test_head_pred.flatten())
        head_mae = np.mean(np.abs(y_test_head_true - y_test_head_pred))

        neck_force_mse = mean_squared_error(y_test_neck_force_true.flatten(), y_test_neck_force_pred.flatten())
        neck_force_mae = np.mean(np.abs(y_test_neck_force_true - y_test_neck_force_pred))

        neck_moment_mse = mean_squared_error(y_test_neck_moment_true.flatten(), y_test_neck_moment_pred.flatten())
        neck_moment_mae = np.mean(np.abs(y_test_neck_moment_true - y_test_neck_moment_pred))

        chest_mse = mean_squared_error(y_test_chest_true.flatten(), y_test_chest_pred.flatten())
        chest_mae = np.mean(np.abs(y_test_chest_true - y_test_chest_pred))

        # 保存结果并计算 ISO 评分
        result_file_path = os.path.join(fold_save_dir, 'result.txt')
        with open(result_file_path, 'w', encoding='utf-8') as f:
            def print_and_write(*args, **kwargs):
                message = ' '.join(str(arg) for arg in args)
                print(*args, **kwargs)
                f.write(message + '\n')
                f.flush()

            print_and_write("\n评估指标:")
            print_and_write(f"HeadAcceleration MSE: {head_mse}, MAE: {head_mae}")
            print_and_write(f"NeckForce MSE: {neck_force_mse}, MAE: {neck_force_mae}")
            print_and_write(f"NeckMoment MSE: {neck_moment_mse}, MAE: {neck_moment_mae}")
            print_and_write(f"ChestDisplacement MSE: {chest_mse}, MAE: {chest_mae}")

            print_and_write("\n计算ISO评分:")
            time_axis = np.linspace(0, 0.2, 100)  # 请根据实际时间轴调整

            # 初始化存储列表
            head_iso_scores = []
            head_corridor_scores = []
            head_phase_scores = []
            head_magnitude_scores = []
            head_slope_scores = []

            neck_force_iso_scores = []
            neck_force_corridor_scores = []
            neck_force_phase_scores = []
            neck_force_magnitude_scores = []
            neck_force_slope_scores = []

            neck_moment_iso_scores = []
            neck_moment_corridor_scores = []
            neck_moment_phase_scores = []
            neck_moment_magnitude_scores = []
            neck_moment_slope_scores = []

            chest_iso_scores = []
            chest_corridor_scores = []
            chest_phase_scores = []
            chest_magnitude_scores = []
            chest_slope_scores = []

            num_samples_to_calc = len(y_test_head_true)
            for i in range(num_samples_to_calc):
                print_and_write(f"\nSample {i + 1} ISO评分:")

                head_score, head_corr, head_pha, head_mag, head_slo = calculate_iso_scores_for_sample(
                    y_test_head_true, y_test_head_pred, time_axis, i, "HeadAcceleration", f)
                head_iso_scores.append(head_score)
                head_corridor_scores.append(head_corr)
                head_phase_scores.append(head_pha)
                head_magnitude_scores.append(head_mag)
                head_slope_scores.append(head_slo)

                nf_score, nf_corr, nf_pha, nf_mag, nf_slo = calculate_iso_scores_for_sample(
                    y_test_neck_force_true, y_test_neck_force_pred, time_axis, i, "NeckForce", f)
                neck_force_iso_scores.append(nf_score)
                neck_force_corridor_scores.append(nf_corr)
                neck_force_phase_scores.append(nf_pha)
                neck_force_magnitude_scores.append(nf_mag)
                neck_force_slope_scores.append(nf_slo)

                nm_score, nm_corr, nm_pha, nm_mag, nm_slo = calculate_iso_scores_for_sample(
                    y_test_neck_moment_true, y_test_neck_moment_pred, time_axis, i, "NeckMoment", f)
                neck_moment_iso_scores.append(nm_score)
                neck_moment_corridor_scores.append(nm_corr)
                neck_moment_phase_scores.append(nm_pha)
                neck_moment_magnitude_scores.append(nm_mag)
                neck_moment_slope_scores.append(nm_slo)

                ch_score, ch_corr, ch_pha, ch_mag, ch_slo = calculate_iso_scores_for_sample(
                    y_test_chest_true, y_test_chest_pred, time_axis, i, "ChestDisplacement", f)
                chest_iso_scores.append(ch_score)
                chest_corridor_scores.append(ch_corr)
                chest_phase_scores.append(ch_pha)
                chest_magnitude_scores.append(ch_mag)
                chest_slope_scores.append(ch_slo)

            def filter_valid_scores(scores):
                return [s for s in scores if s is not None]

            # 计算平均值
            valid_head = filter_valid_scores(head_iso_scores)
            avg_head_iso = np.mean(valid_head) if valid_head else None
            avg_head_corridor = np.mean(filter_valid_scores(head_corridor_scores)) if valid_head else None
            avg_head_phase = np.mean(filter_valid_scores(head_phase_scores)) if valid_head else None
            avg_head_magnitude = np.mean(filter_valid_scores(head_magnitude_scores)) if valid_head else None
            avg_head_slope = np.mean(filter_valid_scores(head_slope_scores)) if valid_head else None

            valid_nf = filter_valid_scores(neck_force_iso_scores)
            avg_neck_force_iso = np.mean(valid_nf) if valid_nf else None
            avg_neck_force_corridor = np.mean(filter_valid_scores(neck_force_corridor_scores)) if valid_nf else None
            avg_neck_force_phase = np.mean(filter_valid_scores(neck_force_phase_scores)) if valid_nf else None
            avg_neck_force_magnitude = np.mean(filter_valid_scores(neck_force_magnitude_scores)) if valid_nf else None
            avg_neck_force_slope = np.mean(filter_valid_scores(neck_force_slope_scores)) if valid_nf else None

            valid_nm = filter_valid_scores(neck_moment_iso_scores)
            avg_neck_moment_iso = np.mean(valid_nm) if valid_nm else None
            avg_neck_moment_corridor = np.mean(filter_valid_scores(neck_moment_corridor_scores)) if valid_nm else None
            avg_neck_moment_phase = np.mean(filter_valid_scores(neck_moment_phase_scores)) if valid_nm else None
            avg_neck_moment_magnitude = np.mean(filter_valid_scores(neck_moment_magnitude_scores)) if valid_nm else None
            avg_neck_moment_slope = np.mean(filter_valid_scores(neck_moment_slope_scores)) if valid_nm else None

            valid_ch = filter_valid_scores(chest_iso_scores)
            avg_chest_iso = np.mean(valid_ch) if valid_ch else None
            avg_chest_corridor = np.mean(filter_valid_scores(chest_corridor_scores)) if valid_ch else None
            avg_chest_phase = np.mean(filter_valid_scores(chest_phase_scores)) if valid_ch else None
            avg_chest_magnitude = np.mean(filter_valid_scores(chest_magnitude_scores)) if valid_ch else None
            avg_chest_slope = np.mean(filter_valid_scores(chest_slope_scores)) if valid_ch else None

            # 输出平均值
            if avg_head_iso is not None:
                print_and_write(f"HeadAcceleration - 平均ISO评分: {avg_head_iso:.2f} (Corridor: {avg_head_corridor:.2f}, Phase: {avg_head_phase:.2f}, Magnitude: {avg_head_magnitude:.2f}, Slope: {avg_head_slope:.2f})")
            else:
                print_and_write("HeadAcceleration - 没有有效的ISO评分")

            if avg_neck_force_iso is not None:
                print_and_write(f"NeckForce - 平均ISO评分: {avg_neck_force_iso:.2f} (Corridor: {avg_neck_force_corridor:.2f}, Phase: {avg_neck_force_phase:.2f}, Magnitude: {avg_neck_force_magnitude:.2f}, Slope: {avg_neck_force_slope:.2f})")
            else:
                print_and_write("NeckForce - 没有有效的ISO评分")

            if avg_neck_moment_iso is not None:
                print_and_write(f"NeckMoment - 平均ISO评分: {avg_neck_moment_iso:.2f} (Corridor: {avg_neck_moment_corridor:.2f}, Phase: {avg_neck_moment_phase:.2f}, Magnitude: {avg_neck_moment_magnitude:.2f}, Slope: {avg_neck_moment_slope:.2f})")
            else:
                print_and_write("NeckMoment - 没有有效的ISO评分")

            if avg_chest_iso is not None:
                print_and_write(f"ChestDisplacement - 平均ISO评分: {avg_chest_iso:.2f} (Corridor: {avg_chest_corridor:.2f}, Phase: {avg_chest_phase:.2f}, Magnitude: {avg_chest_magnitude:.2f}, Slope: {avg_chest_slope:.2f})")
            else:
                print_and_write("ChestDisplacement - 没有有效的ISO评分")

            # 将当前折的评分追加到汇总字典
            all_head_metrics['corridor'].extend([s for s in head_corridor_scores if s is not None])
            all_head_metrics['phase'].extend([s for s in head_phase_scores if s is not None])
            all_head_metrics['magnitude'].extend([s for s in head_magnitude_scores if s is not None])
            all_head_metrics['slope'].extend([s for s in head_slope_scores if s is not None])
            all_head_metrics['overall'].extend([s for s in head_iso_scores if s is not None])

            all_neck_force_metrics['corridor'].extend([s for s in neck_force_corridor_scores if s is not None])
            all_neck_force_metrics['phase'].extend([s for s in neck_force_phase_scores if s is not None])
            all_neck_force_metrics['magnitude'].extend([s for s in neck_force_magnitude_scores if s is not None])
            all_neck_force_metrics['slope'].extend([s for s in neck_force_slope_scores if s is not None])
            all_neck_force_metrics['overall'].extend([s for s in neck_force_iso_scores if s is not None])

            all_neck_moment_metrics['corridor'].extend([s for s in neck_moment_corridor_scores if s is not None])
            all_neck_moment_metrics['phase'].extend([s for s in neck_moment_phase_scores if s is not None])
            all_neck_moment_metrics['magnitude'].extend([s for s in neck_moment_magnitude_scores if s is not None])
            all_neck_moment_metrics['slope'].extend([s for s in neck_moment_slope_scores if s is not None])
            all_neck_moment_metrics['overall'].extend([s for s in neck_moment_iso_scores if s is not None])

            all_chest_metrics['corridor'].extend([s for s in chest_corridor_scores if s is not None])
            all_chest_metrics['phase'].extend([s for s in chest_phase_scores if s is not None])
            all_chest_metrics['magnitude'].extend([s for s in chest_magnitude_scores if s is not None])
            all_chest_metrics['slope'].extend([s for s in chest_slope_scores if s is not None])
            all_chest_metrics['overall'].extend([s for s in chest_iso_scores if s is not None])

            # 将当前折结果存入 all_fold_results（注意在 with 块内）
            fold_result = {
                'fold': fold + 1,
                'head_mse': head_mse,
                'head_mae': head_mae,
                'neck_force_mse': neck_force_mse,
                'neck_force_mae': neck_force_mae,
                'neck_moment_mse': neck_moment_mse,
                'neck_moment_mae': neck_moment_mae,
                'chest_mse': chest_mse,
                'chest_mae': chest_mae,
                'head_iso' : avg_head_iso,
                'neck_force_iso' : avg_neck_force_iso,
                'neck_moment_iso' : avg_neck_moment_iso,
                'chest_iso': avg_chest_iso
            }
            all_fold_results.append(fold_result)

        print(f"\n 第{fold + 1} 折结果已保存至: {result_file_path}")

        # 可视化预测图像
        images_dir = os.path.join(fold_save_dir, 'prediction_images')
        os.makedirs(images_dir, exist_ok=True)
        num_samples_to_plot = min(20, len(y_val_head_raw))
        for i in range(num_samples_to_plot):
            fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

            # HeadAcceleration
            axes[0].plot(y_test_head_true[i], 'g-', label='True', linewidth=1, alpha=0.8)
            axes[0].plot(y_test_head_pred[i], 'r--', label='Predict', linewidth=1, alpha=0.8)
            if i < len(head_iso_scores) and head_iso_scores[i] is not None:
                axes[0].text(0.02, 0.98, f'ISO: {head_iso_scores[i]:.2f}',
                             transform=axes[0].transAxes, fontsize=10,
                             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            axes[0].set_title(f'HeadAcceleration - Sample {i + 1}')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # NeckForce
            axes[1].plot(y_test_neck_force_true[i], 'g-', label='True', linewidth=1, alpha=0.8)
            axes[1].plot(y_test_neck_force_pred[i], 'r--', label='Predict', linewidth=1, alpha=0.8)
            if i < len(neck_force_iso_scores) and neck_force_iso_scores[i] is not None:
                axes[1].text(0.02, 0.98, f'ISO: {neck_force_iso_scores[i]:.2f}',
                             transform=axes[1].transAxes, fontsize=10,
                             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            axes[1].set_title(f'NeckForce - Sample {i + 1}')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            # NeckMoment
            axes[2].plot(y_test_neck_moment_true[i], 'g-', label='True', linewidth=1, alpha=0.8)
            axes[2].plot(y_test_neck_moment_pred[i], 'r--', label='Predict', linewidth=1, alpha=0.8)
            if i < len(neck_moment_iso_scores) and neck_moment_iso_scores[i] is not None:
                axes[2].text(0.02, 0.98, f'ISO: {neck_moment_iso_scores[i]:.2f}',
                             transform=axes[2].transAxes, fontsize=10,
                             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            axes[2].set_title(f'NeckMoment - Sample {i + 1}')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

            # ChestDisplacement
            axes[3].plot(y_test_chest_true[i], 'g-', label='True', linewidth=1, alpha=0.8)
            axes[3].plot(y_test_chest_pred[i], 'r--', label='Predict', linewidth=1, alpha=0.8)
            if i < len(chest_iso_scores) and chest_iso_scores[i] is not None:
                axes[3].text(0.02, 0.98, f'ISO: {chest_iso_scores[i]:.2f}',
                             transform=axes[3].transAxes, fontsize=10,
                             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            axes[3].set_title(f'ChestDisplacement - Sample {i + 1}')
            axes[3].set_xlabel('Time Steps')
            axes[3].legend()
            axes[3].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(images_dir, f'prediction_sample_{i + 1}.png'), dpi=300, bbox_inches='tight')
            plt.close()

        # 清理内存
        K.clear_session()
        tf.compat.v1.reset_default_graph()
        del model, history
        print(f"\n 第 {fold + 1} 折训练完成，内存已清理")

    # 绘制所有折的 ISO 评分分布直方图
    all_detailed_metrics = [all_head_metrics, all_neck_force_metrics, all_neck_moment_metrics, all_chest_metrics]
    plot_metrics_histograms(all_detailed_metrics, save_dir)
    # 汇总所有折的结果
    print("5折交叉验证全部完成！总结果汇总")
    results_df = pd.DataFrame(all_fold_results)
    summary = results_df.agg(['mean', 'std']).round(4)
    summary_path = os.path.join(save_dir, '5fold_summary.csv')
    summary.to_csv(summary_path, encoding='utf-8-sig')
    print(f"\n5折汇总结果已保存至: {summary_path}")
    print("\n平均指标:")
    print(summary.loc['mean'])
    print("\n标准差:")
    print(summary.loc['std'])
    return all_fold_results
if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    if gpus :
        try :
            # 设置仅允许按需增长显存，避免一次性全部占用
            for gpu in gpus :
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e :
            print(e)
    all_fold_results = main()

