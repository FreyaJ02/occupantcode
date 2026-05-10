#归一化处理参考PPT
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
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, Callback
from keras.models import load_model
from keras.layers import Bidirectional
import tensorflow as tf
from keras import backend as K
from objective_rating_metrics.rating import ISO18571

# 设置随机种子以确保可复现性
np.random.seed(42)
tf.random.set_seed(42)


def define_hybrid_model(input_length=100, scalar_length=3, output_length=100):
    """
    定义混合模型：LSTM处理时序数据 + Dense处理标量参数
    仅预测颈部力矩曲线（NeckMoment）
    注意：输出层使用线性激活，模型输出为归一化后的值(-1,1)，需要反归一化才能得到原始值
    """
    # 输入层
    input_lstm = Input(shape=(input_length, 1), name='lstm_input')  # CrashPulse时序数据输入
    input_dense = Input(shape=(scalar_length,), name='dense_input')  # 标量参数输入(Airbag, Gender, Seatbelt)

    # LSTM分支
    lstm_branch = Bidirectional(LSTM(units=32, return_sequences=True))(input_lstm)
    lstm_branch = Dropout(0.2)(lstm_branch)
    lstm_branch = Bidirectional(LSTM(units=16))(lstm_branch)
    lstm_branch = Dropout(0.2)(lstm_branch)

    # Dense分支处理标量参数
    dense_branch = Dense(16, activation='relu')(input_dense)
    dense_branch = Dropout(0.2)(dense_branch)

    # 合并两个分支
    merged = concatenate([lstm_branch, dense_branch])

    # 共享的全连接层
    shared = Dense(64, activation='relu')(merged)
    shared = Dropout(0.2)(shared)
    shared = Dense(32, activation='relu')(shared)
    shared = Dropout(0.2)(shared)

    # 单个输出分支 - 仅颈部力矩
    neck_moment_output = Dense(output_length, name='neck_moment_output')(shared)  # NeckMoment

    # 创建单输出模型
    model = Model(inputs=[input_lstm, input_dense],
                  outputs=neck_moment_output)

    # 编译模型，仅使用颈部力矩的损失
    model.compile(
        loss='mse',
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3)
    )
    return model


def load_filtered_npy_data(data_dir):
    """
    从npy文件加载所有数据，与main_injury_informer.py保持一致
    返回的数据为原始值，后续需要进行归一化处理
    """
    # 加载标量数据
    airbag = np.load(os.path.join(data_dir, 'Airbag.npy'))
    gender = np.load(os.path.join(data_dir, 'Gender.npy'))
    seatbelt = np.load(os.path.join(data_dir, 'Seatbelt.npy'))

    # 加载时序输入数据
    crash_pulse = np.load(os.path.join(data_dir, 'CrashPulse.npy'))

    # 加载时序输出数据 - 仅颈部力矩
    neck_moment = np.load(os.path.join(data_dir, 'NeckMoment.npy'))

    # 检查数据维度是否一致
    assert airbag.shape[0] == gender.shape[0] == seatbelt.shape[0] == \
           crash_pulse.shape[0] == neck_moment.shape[0], "所有数据文件的第一维度必须一致"

    # 组装标量数据
    scalar_data = np.column_stack([airbag, gender, seatbelt])

    return scalar_data, crash_pulse, neck_moment


def cosine_annealing_scheduler(total_epochs, lr_max=1e-3, lr_min=1e-6):
    """
    构造余弦退火学习率调度函数
    """

    def scheduler(epoch, lr):
        if total_epochs <= 1:
            return lr_min
        cosine_decay = 0.5 * (1 + np.cos(np.pi * epoch / (total_epochs - 1)))
        return lr_min + (lr_max - lr_min) * cosine_decay

    return scheduler


def train_model(model, X_train_lstm, X_train_dense, y_train_neck_moment, X_val_lstm, X_val_dense,
                y_val_neck_moment, save_dir):
    """
    训练模型
    注意：y_train_neck_moment 和 y_val_neck_moment 均为归一化后的值(-1,1)
    """
    class EpochTimingCallback(Callback):
        """记录每个epoch及整体训练耗时。"""
        def on_train_begin(self, logs=None):
            self.train_start_time = time.perf_counter()
            self.epoch_times = []
            self.epoch_start_time = None

        def on_epoch_begin(self, epoch, logs=None):
            self.epoch_start_time = time.perf_counter()

        def on_epoch_end(self, epoch, logs=None):
            if self.epoch_start_time is None:
                return
            elapsed = time.perf_counter() - self.epoch_start_time
            self.epoch_times.append(elapsed)
            print(f"Epoch {epoch + 1} 用时: {elapsed:.2f} 秒")

        def on_train_end(self, logs=None):
            self.total_train_time = time.perf_counter() - self.train_start_time
            print(f"训练总用时: {self.total_train_time:.2f} 秒")

    total_epochs = 200
    lr_scheduler = LearningRateScheduler(
        cosine_annealing_scheduler(total_epochs, lr_max=1e-3, lr_min=1e-5),
        verbose=1
    )

    # 模型检查点
    model_path = os.path.join(save_dir, 'best_opendata_lstm_model.h5')
    checkpoint = ModelCheckpoint(
        model_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    timing_callback = EpochTimingCallback()

    # 训练模型
    history = model.fit(
        [X_train_lstm, X_train_dense],
        y_train_neck_moment,
        epochs=total_epochs,
        batch_size=16,
        validation_data=([X_val_lstm, X_val_dense],
                         y_val_neck_moment),
        callbacks=[lr_scheduler, checkpoint, timing_callback],
        verbose=1
    )

    # 将时间统计写入history，方便后续可视化或保存
    history.history['epoch_time_seconds'] = timing_callback.epoch_times
    history.history['total_train_time_seconds'] = [timing_callback.total_train_time]

    # 保存每个epoch的loss记录（按epoch逐行构建，避免不同key长度不一致导致报错）
    train_loss_list = history.history.get('loss', [])
    val_loss_list = history.history.get('val_loss', [])
    epoch_time_list = timing_callback.epoch_times

    num_epochs_logged = len(train_loss_list)
    loss_rows = []
    for i in range(num_epochs_logged):
        loss_rows.append({
            'epoch': i + 1,
            'loss': train_loss_list[i] if i < len(train_loss_list) else np.nan,
            'val_loss': val_loss_list[i] if i < len(val_loss_list) else np.nan,
            'epoch_time_seconds': epoch_time_list[i] if i < len(epoch_time_list) else np.nan
        })

    loss_log_df = pd.DataFrame(loss_rows)
    loss_log_path = os.path.join(save_dir, 'epoch_loss_history.csv')
    loss_log_df.to_csv(loss_log_path, index=False, encoding='utf-8')
    print(f"每个Epoch的loss记录已保存至: {loss_log_path}")

    # 保存训练时间报告
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
    """
    绘制并保存各个输出的 loss 曲线
    """
    os.makedirs(save_dir, exist_ok=True)
    history_dict = history.history

    # 定义所有输出名称
    output_names = ['neck_moment_output']
    colors = ['b']

    # 绘制每个输出的loss曲线
    for i, output_name in enumerate(output_names):
        train_loss_key = f'{output_name}_loss'
        val_loss_key = f'val_{output_name}_loss'

        if train_loss_key in history_dict:
            plt.figure(figsize=(10, 5))
            plt.plot(history_dict[train_loss_key], f'{colors[i]}-', label=f'{output_name} Training Loss', linewidth=2)
            if val_loss_key in history_dict:
                plt.plot(history_dict[val_loss_key], f'{colors[i]}--', label=f'{output_name} Validation Loss',
                         linewidth=2)
            plt.title(f'{output_name} Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.yscale('log')

            loss_plot_path = os.path.join(save_dir, f'{output_name}_training_loss.png')
            plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"{output_name}损失曲线图已保存至: {loss_plot_path}")

    # 绘制总loss曲线
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


def plot_iso_scores_histograms(detailed_metrics, save_dir):
    """
    绘制ISO评分分布直方图
    """
    os.makedirs(save_dir, exist_ok=True)

    # 过滤掉None值
    def filter_valid_scores(scores):
        return [x for x in scores if x is not None]

    plt.figure(figsize=(18, 8))

    metric_keys = ['corridor', 'phase', 'magnitude', 'slope', 'overall']
    colors = ['skyblue', 'lightgreen', 'salmon', 'gold', 'orchid']

    for idx, key in enumerate(metric_keys):
        plt.subplot(2, 3, idx + 1)
        scores = filter_valid_scores(detailed_metrics[key])
        if scores:
            plt.hist(scores, bins=20, color=colors[idx], edgecolor='black', alpha=0.7)
        plt.title(f'{key.capitalize()} Score')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图像
    iso_histogram_path = os.path.join(save_dir, 'iso_scores_histograms.png')
    plt.savefig(iso_histogram_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"ISO评分分布直方图已保存至: {iso_histogram_path}")
    return iso_histogram_path


def calculate_iso_scores_for_sample(y_true, y_pred, time_axis, sample_idx, output_name, f):
    """
    计算单个样本的ISO评分
    """

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

        # 记录详细指标
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
    # 数据目录 - 使用未过滤的数据（参考main_injury_informer.py）
    data_dir = '/public/home/acnebvshuq/js/occupant/dataset/final_dataset_4547'
    save_dir = '/public/home/acnebvshuq/js/occupant/norm/result/mse/neckmoment'
    os.makedirs(save_dir, exist_ok=True)

    # 加载数据
    print("加载数据...")
    scalar_data, crash_pulse, neck_moment = load_filtered_npy_data(data_dir)

    # 检查数据形状
    print(f"标量数据形状: {scalar_data.shape}")
    print(f"CrashPulse形状: {crash_pulse.shape}")
    print(f"NeckMoment形状: {neck_moment.shape}")

    # 检查数据形状是否匹配
    total_samples = scalar_data.shape[0]
    assert scalar_data.shape[0] == crash_pulse.shape[0] == neck_moment.shape[0], \
           "所有数据的样本数量必须一致"

    # 对CrashPulse时序输入数据进行归一化
    scaler_crash = MinMaxScaler()
    
    # 对颈部力矩输出数据进行归一化(-1,1)
    scaler_neck_moment = MinMaxScaler(feature_range=(-1, 1))
    
    # 标量数据不进行归一化，直接使用原始值
    X_scalar = scalar_data

    # 归一化时序输入数据(CrashPulse)
    # 重塑CrashPulse为(total_samples*100, 1)用于归一化
    crash_pulse_reshaped = crash_pulse.reshape(-1, 1)
    X_crash = scaler_crash.fit_transform(crash_pulse_reshaped).reshape(crash_pulse.shape)

    # 归一化颈部力矩输出数据(-1,1)
    # 重塑为(total_samples*100, 1)用于归一化
    neck_moment_reshaped = neck_moment.reshape(-1, 1)
    y_neck_moment_normalized = scaler_neck_moment.fit_transform(neck_moment_reshaped).reshape(neck_moment.shape)

    # 重塑CrashPulse为(total_samples, 100, 1)以适应LSTM输入
    X_crash = X_crash.reshape((X_crash.shape[0], X_crash.shape[1], 1))

    # 划分训练集、验证集和测试集 (80%, 10%, 10%)
    indices = np.random.permutation(total_samples)

    train_end = int(0.8 * total_samples)
    val_end = int(0.9 * total_samples)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    # 训练集 - 输出数据使用归一化后的值(-1,1)
    X_train_crash = X_crash[train_indices]
    X_train_scalar = X_scalar[train_indices]
    y_train_neck_moment = y_neck_moment_normalized[train_indices]

    # 验证集 - 输出数据使用归一化后的值(-1,1)
    X_val_crash = X_crash[val_indices]
    X_val_scalar = X_scalar[val_indices]
    y_val_neck_moment = y_neck_moment_normalized[val_indices]

    # 测试集 - 保存原始值和归一化值，用于后续评估
    X_test_crash = X_crash[test_indices]
    X_test_scalar = X_scalar[test_indices]
    y_test_neck_moment_normalized = y_neck_moment_normalized[test_indices]  # 归一化值，用于模型评估
    y_test_neck_moment_original = neck_moment[test_indices]  # 原始值，用于ISO评分计算

    print(f"\n数据集划分完成:")
    print(f"训练集样本数: {len(train_indices)}")
    print(f"验证集样本数: {len(val_indices)}")
    print(f"测试集样本数: {len(test_indices)}")
    
    # 保存颈部力矩的scaler，用于后续反归一化
    import joblib
    scaler_path = os.path.join(save_dir, 'neck_moment_scaler.pkl')
    joblib.dump(scaler_neck_moment, scaler_path)
    print(f"颈部力矩归一化器已保存至: {scaler_path}")

    # 构建模型
    model = define_hybrid_model(input_length=100, scalar_length=3, output_length=100)
    model.summary()

    # 训练模型
    print("开始训练模型...")
    history = train_model(
        model,
        X_train_crash, X_train_scalar,
        y_train_neck_moment,
        X_val_crash, X_val_scalar,
        y_val_neck_moment,
        save_dir
    )

    # 绘制训练损失图像并保存
    loss_plot_path = plot_training_history(history, save_dir)

    # 加载保存的最佳模型进行测试
    best_model_path = os.path.join(save_dir, 'best_opendata_lstm_model.h5')
    if os.path.exists(best_model_path):
        print(f"\n加载最佳模型: {best_model_path}")
        model = load_model(best_model_path)
        # 重新编译模型
        model.compile(
            loss='mse',
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3)
        )
        print("已成功加载最佳模型，将在测试集上评估")
    else:
        print(f"\n警告: 未找到最佳模型文件 {best_model_path}，使用当前内存中的模型进行评估")

    # 评估模型 - 使用归一化值进行评估
    result = model.evaluate([X_test_crash, X_test_scalar],
                            y_test_neck_moment_normalized)
    print(f"Test Losses: {result}")

    # 预测 - 输出为归一化值(-1,1)，需要反归一化
    y_pred_neck_moment_normalized = model.predict([X_test_crash, X_test_scalar])

    # 反归一化：将预测值和真实值从(-1,1)恢复到原始值范围
    # 重塑为(-1, 1)以进行反归一化
    y_pred_neck_flat = y_pred_neck_moment_normalized.reshape(-1, 1)
    y_test_neck_norm_flat = y_test_neck_moment_normalized.reshape(-1, 1)
    
    # 反归一化
    y_pred_neck_original = scaler_neck_moment.inverse_transform(y_pred_neck_flat).reshape(y_pred_neck_moment_normalized.shape)
    y_test_neck_from_norm = scaler_neck_moment.inverse_transform(y_test_neck_norm_flat).reshape(y_test_neck_moment_normalized.shape)
    
    # 使用反归一化后的值进行评估和可视化
    y_test_neck_moment_true = y_test_neck_moment_original  # 原始真实值
    y_test_neck_moment_pred = y_pred_neck_original  # 反归一化后的预测值

    # 计算评估指标
    neck_moment_mse = mean_squared_error(y_test_neck_moment_true.flatten(), y_test_neck_moment_pred.flatten())
    neck_moment_mae = np.mean(np.abs(y_test_neck_moment_true - y_test_neck_moment_pred))

    # 保存结果
    result_file_path = os.path.join(save_dir, 'result.txt')
    with open(result_file_path, 'w', encoding='utf-8') as f:
        def print_and_write(*args, **kwargs):
            message = ' '.join(str(arg) for arg in args)
            print(*args, **kwargs)
            f.write(message + '\n')
            f.flush()

        print_and_write("\n评估指标:")
        print_and_write(f"NeckMoment MSE: {neck_moment_mse}, MAE: {neck_moment_mae}")

        # 计算ISO评分
        print_and_write("\n计算ISO评分:")
        time_axis = np.linspace(0, 0.2, 100)  # 假设时间轴为0-0.2秒，100个点

        # 初始化所有输出的评分存储
        neck_moment_iso_scores = []
        neck_moment_corridor_scores = []
        neck_moment_phase_scores = []
        neck_moment_magnitude_scores = []
        neck_moment_slope_scores = []

        # 计算每个样本的ISO评分
        num_samples_to_calc = len(y_test_neck_moment_true)  # 计算所有测试样本

        for i in range(num_samples_to_calc):
            print_and_write(f"\nSample {i + 1} ISO评分:")

            # 计算NeckMoment的ISO评分
            neck_moment_score, neck_moment_corridor, neck_moment_phase, neck_moment_magnitude, neck_moment_slope = calculate_iso_scores_for_sample(
                y_test_neck_moment_true, y_test_neck_moment_pred, time_axis, i, "NeckMoment", f)
            neck_moment_iso_scores.append(neck_moment_score)
            neck_moment_corridor_scores.append(neck_moment_corridor)
            neck_moment_phase_scores.append(neck_moment_phase)
            neck_moment_magnitude_scores.append(neck_moment_magnitude)
            neck_moment_slope_scores.append(neck_moment_slope)

        # 计算所有测试集的平均ISO评分
        print_and_write(f"\n所有测试集的平均ISO评分:")

        # 过滤有效评分
        def filter_valid_scores(scores):
            return [s for s in scores if s is not None]

        # 计算NeckMoment的平均评分
        valid_neck_moment_scores = filter_valid_scores(neck_moment_iso_scores)
        if valid_neck_moment_scores:
            avg_neck_moment_iso = np.mean(valid_neck_moment_scores)
            avg_neck_moment_corridor = np.mean(filter_valid_scores(neck_moment_corridor_scores))
            avg_neck_moment_phase = np.mean(filter_valid_scores(neck_moment_phase_scores))
            avg_neck_moment_magnitude = np.mean(filter_valid_scores(neck_moment_magnitude_scores))
            avg_neck_moment_slope = np.mean(filter_valid_scores(neck_moment_slope_scores))
            print_and_write(
                f"NeckMoment - 平均ISO评分: {avg_neck_moment_iso:.2f} (Corridor: {avg_neck_moment_corridor:.2f}, Phase: {avg_neck_moment_phase:.2f}, Magnitude: {avg_neck_moment_magnitude:.2f}, Slope: {avg_neck_moment_slope:.2f})")
        else:
            print_and_write("NeckMoment - 没有有效的ISO评分")

        # 将五个ISO指标统一可视化为一张分布直方图
        detailed_metrics = {
            'corridor': neck_moment_corridor_scores,
            'phase': neck_moment_phase_scores,
            'magnitude': neck_moment_magnitude_scores,
            'slope': neck_moment_slope_scores,
            'overall': neck_moment_iso_scores
        }
        iso_histogram_path = plot_iso_scores_histograms(detailed_metrics, save_dir)
        print_and_write(f"ISO评分分布直方图已保存至: {iso_histogram_path}")

    print(f"\n结果已保存至: {result_file_path}")

    # 可视化预测结果 - 使用原始值进行可视化
    images_dir = os.path.join(save_dir, 'prediction_images')
    os.makedirs(images_dir, exist_ok=True)

    num_samples_to_plot = min(20, len(y_test_neck_moment_true))
    for i in range(num_samples_to_plot):
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        # NeckMoment
        ax.plot(y_test_neck_moment_true[i], 'g-', label='True', linewidth=1, alpha=0.8)
        ax.plot(y_test_neck_moment_pred[i], 'r--', label='Predict', linewidth=1, alpha=0.8)

        # 添加ISO评分标注
        if i < len(neck_moment_iso_scores) and neck_moment_iso_scores[i] is not None:
            ax.text(0.02, 0.98, f'ISO Score: {neck_moment_iso_scores[i]:.2f}',
                     transform=ax.transAxes, fontsize=10,
                     verticalalignment='top', horizontalalignment='left',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_title(f'NeckMoment - Sample {i + 1}')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Neck Moment')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        image_path = os.path.join(images_dir, f'prediction_sample_{i + 1}.png')
        plt.savefig(image_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"预测结果图已保存至: {image_path}")

    return model, None


if __name__ == "__main__":
    model, _ = main()