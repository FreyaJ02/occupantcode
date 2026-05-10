import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras import Model
from keras.layers import Input, Dense, Dropout, concatenate, LSTM, Bidirectional
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, Callback

import tensorflow as tf
from objective_rating_metrics.rating import ISO18571

from isoloss_new import ISO18571Loss_DTW


np.random.seed(42)
tf.random.set_seed(42)


def define_hybrid_model(input_length=100, scalar_length=3, output_length=100):
    input_lstm = Input(shape=(input_length, 1), name='lstm_input')
    input_dense = Input(shape=(scalar_length,), name='dense_input')

    lstm_branch = Bidirectional(LSTM(units=32, return_sequences=True))(input_lstm)
    lstm_branch = Dropout(0.2)(lstm_branch)
    lstm_branch = Bidirectional(LSTM(units=16))(lstm_branch)
    lstm_branch = Dropout(0.2)(lstm_branch)

    dense_branch = Dense(16, activation='relu')(input_dense)
    dense_branch = Dropout(0.2)(dense_branch)

    merged = concatenate([lstm_branch, dense_branch])

    shared = Dense(64, activation='relu')(merged)
    shared = Dropout(0.2)(shared)
    shared = Dense(32, activation='relu')(shared)
    shared = Dropout(0.2)(shared)

    neck_force_output = Dense(output_length, name='neck_force_output')(shared)

    model = Model(inputs=[input_lstm, input_dense], outputs=neck_force_output)
    return model


def load_filtered_npy_data(data_dir):
    airbag = np.load(os.path.join(data_dir, 'Airbag.npy'))
    gender = np.load(os.path.join(data_dir, 'Gender.npy'))
    seatbelt = np.load(os.path.join(data_dir, 'Seatbelt.npy'))

    crash_pulse = np.load(os.path.join(data_dir, 'CrashPulse.npy'))
    neck_force = np.load(os.path.join(data_dir, 'NeckForce.npy'))

    assert airbag.shape[0] == gender.shape[0] == seatbelt.shape[0] == crash_pulse.shape[0] == neck_force.shape[0], \
        'All data files must have the same number of samples.'

    scalar_data = np.column_stack([airbag, gender, seatbelt])
    return scalar_data, crash_pulse, neck_force


def cosine_annealing_scheduler(total_epochs, lr_max=1e-3, lr_min=1e-6):
    def scheduler(epoch, lr):
        if total_epochs <= 1:
            return lr_min
        cosine_decay = 0.5 * (1 + np.cos(np.pi * epoch / (total_epochs - 1)))
        return lr_min + (lr_max - lr_min) * cosine_decay

    return scheduler


class EpochTimingCallback(Callback):
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
        print(f"{self.stage_name} - Epoch {epoch + 1} time: {elapsed:.2f}s")

    def on_train_end(self, logs=None):
        self.total_train_time = time.perf_counter() - self.train_start_time
        print(f"{self.stage_name} total time: {self.total_train_time:.2f}s")


def train_model_two_stage(model,
                          X_train_lstm, X_train_dense, y_train_neck_force,
                          X_val_lstm, X_val_dense, y_val_neck_force,
                          save_dir):
    os.makedirs(save_dir, exist_ok=True)

    stage1_epochs = 100
    stage2_epochs = 100
    stage1_weights_path = os.path.join(save_dir, 'stage1_mse_best.weights.h5')
    stage2_weights_path = os.path.join(save_dir, 'stage2_isoloss_best.weights.h5')

    print('\n' + '=' * 60)
    print('Stage 1: MSE pretraining')
    print('=' * 60)

    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))

    stage1_timing = EpochTimingCallback(stage_name='Stage 1 MSE')
    history_stage1 = model.fit(
        [X_train_lstm, X_train_dense],
        y_train_neck_force,
        epochs=stage1_epochs,
        batch_size=16,
        validation_data=([X_val_lstm, X_val_dense], y_val_neck_force),
        callbacks=[
            LearningRateScheduler(cosine_annealing_scheduler(stage1_epochs, lr_max=1e-3, lr_min=1e-5), verbose=1),
            ModelCheckpoint(stage1_weights_path, monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1),
            stage1_timing,
        ],
        verbose=1,
    )

    if os.path.exists(stage1_weights_path):
        model.load_weights(stage1_weights_path)

    print('\n' + '=' * 60)
    print('Stage 2: ISO18571Loss_DTW finetuning')
    print('=' * 60)

    model.compile(loss=ISO18571Loss_DTW(), optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))

    stage2_timing = EpochTimingCallback(stage_name='Stage 2 ISOLOSS')
    history_stage2 = model.fit(
        [X_train_lstm, X_train_dense],
        y_train_neck_force,
        epochs=stage2_epochs,
        batch_size=16,
        validation_data=([X_val_lstm, X_val_dense], y_val_neck_force),
        callbacks=[
            LearningRateScheduler(cosine_annealing_scheduler(stage2_epochs, lr_max=1e-4, lr_min=1e-6), verbose=1),
            ModelCheckpoint(stage2_weights_path, monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1),
            stage2_timing,
        ],
        verbose=1,
    )

    if os.path.exists(stage2_weights_path):
        model.load_weights(stage2_weights_path)

    rows = []
    s1_loss = history_stage1.history.get('loss', [])
    s1_val = history_stage1.history.get('val_loss', [])
    s2_loss = history_stage2.history.get('loss', [])
    s2_val = history_stage2.history.get('val_loss', [])

    for i, v in enumerate(s1_loss):
        rows.append({
            'stage': 'stage1_mse',
            'epoch': i + 1,
            'global_epoch': i + 1,
            'loss': v,
            'val_loss': s1_val[i] if i < len(s1_val) else np.nan,
            'epoch_time_seconds': stage1_timing.epoch_times[i] if i < len(stage1_timing.epoch_times) else np.nan,
        })

    offset = len(s1_loss)
    for i, v in enumerate(s2_loss):
        rows.append({
            'stage': 'stage2_isoloss',
            'epoch': i + 1,
            'global_epoch': offset + i + 1,
            'loss': v,
            'val_loss': s2_val[i] if i < len(s2_val) else np.nan,
            'epoch_time_seconds': stage2_timing.epoch_times[i] if i < len(stage2_timing.epoch_times) else np.nan,
        })

    pd.DataFrame(rows).to_csv(
        os.path.join(save_dir, 'two_stage_epoch_loss_history.csv'),
        index=False,
        encoding='utf-8',
    )

    with open(os.path.join(save_dir, 'two_stage_training_time_summary.txt'), 'w', encoding='utf-8') as f:
        f.write(f'Stage 1 total time: {stage1_timing.total_train_time:.4f}s\n')
        f.write(f'Stage 2 total time: {stage2_timing.total_train_time:.4f}s\n')
        f.write(f'Total time: {stage1_timing.total_train_time + stage2_timing.total_train_time:.4f}s\n')

    return model, history_stage1, history_stage2, stage1_weights_path, stage2_weights_path


def set_log_scale_if_positive(values):
    vals = [v for v in values if np.isfinite(v)]
    if vals and np.min(vals) > 0:
        plt.yscale('log')


def plot_two_stage_training_history(history_stage1, history_stage2, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    stage1_loss = history_stage1.history.get('loss', [])
    stage1_val_loss = history_stage1.history.get('val_loss', [])
    stage2_loss = history_stage2.history.get('loss', [])
    stage2_val_loss = history_stage2.history.get('val_loss', [])

    plt.figure(figsize=(10, 5))
    plt.plot(stage1_loss, label='Stage 1 Train MSE')
    plt.plot(stage1_val_loss, label='Stage 1 Val MSE')
    plt.title('Stage 1 Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    set_log_scale_if_positive(stage1_loss + stage1_val_loss)
    plt.savefig(os.path.join(save_dir, 'stage1_mse_training_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(stage2_loss, label='Stage 2 Train ISO')
    plt.plot(stage2_val_loss, label='Stage 2 Val ISO')
    plt.title('Stage 2 Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    set_log_scale_if_positive(stage2_loss + stage2_val_loss)
    plt.savefig(os.path.join(save_dir, 'stage2_isoloss_training_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()

    x1 = list(range(1, len(stage1_loss) + 1))
    x2 = list(range(len(stage1_loss) + 1, len(stage1_loss) + len(stage2_loss) + 1))

    plt.figure(figsize=(12, 6))
    plt.plot(x1, stage1_loss, label='Stage 1 Train')
    plt.plot(x1, stage1_val_loss, label='Stage 1 Val')
    plt.plot(x2, stage2_loss, label='Stage 2 Train')
    plt.plot(x2, stage2_val_loss, label='Stage 2 Val')
    plt.axvline(x=len(stage1_loss), color='gray', linestyle='--', label='Stage Switch')
    plt.title('Two-Stage Loss Overview')
    plt.xlabel('Global Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    set_log_scale_if_positive(stage1_loss + stage1_val_loss + stage2_loss + stage2_val_loss)
    plt.savefig(os.path.join(save_dir, 'two_stage_training_loss_overview.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_iso_scores_histograms(detailed_metrics, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    def valid(scores):
        return [x for x in scores if x is not None]

    plt.figure(figsize=(18, 8))
    metric_keys = ['corridor', 'phase', 'magnitude', 'slope', 'overall']
    colors = ['skyblue', 'lightgreen', 'salmon', 'gold', 'orchid']

    for idx, key in enumerate(metric_keys):
        plt.subplot(2, 3, idx + 1)
        scores = valid(detailed_metrics[key])
        if scores:
            plt.hist(scores, bins=20, color=colors[idx], edgecolor='black', alpha=0.7)
        plt.title(f'{key.capitalize()} Score')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    iso_histogram_path = os.path.join(save_dir, 'iso_scores_histograms.png')
    plt.savefig(iso_histogram_path, dpi=300, bbox_inches='tight')
    plt.close()
    return iso_histogram_path


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
            f"{output_name} - Sample {sample_idx + 1} - ISO Score: {score:.2f} "
            f"(Corridor: {corridor:.2f}, Phase: {phase:.2f}, Magnitude: {magnitude:.2f}, Slope: {slope:.2f})"
        )

        return score, corridor, phase, magnitude, slope
    except Exception as e:
        print_and_write(f"{output_name} - Sample {sample_idx + 1} - Error calculating ISO score: {e}")
        return None, None, None, None, None


def main():
    data_dir = '/public/home/acnebvshuq/js/occupant/dataset/final_dataset_4547'
    save_dir = '/public/home/acnebvshuq/js/occupant/norm/result/mix/neckforce'
    os.makedirs(save_dir, exist_ok=True)

    print('Loading data...')
    scalar_data, crash_pulse, neck_force = load_filtered_npy_data(data_dir)

    total_samples = scalar_data.shape[0]
    assert scalar_data.shape[0] == crash_pulse.shape[0] == neck_force.shape[0], 'Sample count mismatch.'

    scaler_crash = MinMaxScaler()
    scaler_neck_force = MinMaxScaler(feature_range=(-1, 1))

    X_scalar = scalar_data

    X_crash = scaler_crash.fit_transform(crash_pulse.reshape(-1, 1)).reshape(crash_pulse.shape)
    y_neck_force_normalized = scaler_neck_force.fit_transform(neck_force.reshape(-1, 1)).reshape(neck_force.shape)

    X_crash = X_crash.reshape((X_crash.shape[0], X_crash.shape[1], 1))

    indices = np.random.permutation(total_samples)
    train_end = int(0.8 * total_samples)
    val_end = int(0.9 * total_samples)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    X_train_crash = X_crash[train_indices]
    X_train_scalar = X_scalar[train_indices]
    y_train_neck_force = y_neck_force_normalized[train_indices]

    X_val_crash = X_crash[val_indices]
    X_val_scalar = X_scalar[val_indices]
    y_val_neck_force = y_neck_force_normalized[val_indices]

    X_test_crash = X_crash[test_indices]
    X_test_scalar = X_scalar[test_indices]
    y_test_neck_force_normalized = y_neck_force_normalized[test_indices]
    y_test_neck_force_original = neck_force[test_indices]

    model = define_hybrid_model(input_length=100, scalar_length=3, output_length=100)
    model.summary()

    model, history_stage1, history_stage2, stage1_weights_path, stage2_weights_path = train_model_two_stage(
        model,
        X_train_crash,
        X_train_scalar,
        y_train_neck_force,
        X_val_crash,
        X_val_scalar,
        y_val_neck_force,
        save_dir,
    )

    plot_two_stage_training_history(history_stage1, history_stage2, save_dir)

    model.compile(loss=ISO18571Loss_DTW(), optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))
    result = model.evaluate([X_test_crash, X_test_scalar], y_test_neck_force_normalized, verbose=1)
    print(f'Test Loss: {result}')

    y_pred_neck_force_normalized = model.predict([X_test_crash, X_test_scalar], verbose=1)

    y_pred_neck_original = scaler_neck_force.inverse_transform(
        y_pred_neck_force_normalized.reshape(-1, 1)
    ).reshape(y_pred_neck_force_normalized.shape)

    y_test_neck_force_true = y_test_neck_force_original
    y_test_neck_force_pred = y_pred_neck_original

    neck_force_mse = mean_squared_error(y_test_neck_force_true.flatten(), y_test_neck_force_pred.flatten())
    neck_force_mae = np.mean(np.abs(y_test_neck_force_true - y_test_neck_force_pred))

    result_file_path = os.path.join(save_dir, 'result.txt')
    with open(result_file_path, 'w', encoding='utf-8') as f:
        def print_and_write(*args, **kwargs):
            message = ' '.join(str(arg) for arg in args)
            print(*args, **kwargs)
            f.write(message + '\n')
            f.flush()

        print_and_write('\nTwo-stage training settings:')
        print_and_write('Stage 1 loss: MSE')
        print_and_write('Stage 2 loss: ISO18571Loss_DTW')
        print_and_write(f'Stage 1 best weights: {stage1_weights_path}')
        print_and_write(f'Stage 2 best weights: {stage2_weights_path}')

        print_and_write('\nMetrics:')
        print_and_write(f'NeckForce MSE: {neck_force_mse}, MAE: {neck_force_mae}')

        print_and_write('\nISO scores:')
        time_axis = np.linspace(0, 0.2, 100)

        neck_force_iso_scores = []
        neck_force_corridor_scores = []
        neck_force_phase_scores = []
        neck_force_magnitude_scores = []
        neck_force_slope_scores = []

        for i in range(len(y_test_neck_force_true)):
            print_and_write(f'\nSample {i + 1} ISO score:')
            score, corridor, phase, magnitude, slope = calculate_iso_scores_for_sample(
                y_test_neck_force_true,
                y_test_neck_force_pred,
                time_axis,
                i,
                'NeckForce',
                f,
            )
            neck_force_iso_scores.append(score)
            neck_force_corridor_scores.append(corridor)
            neck_force_phase_scores.append(phase)
            neck_force_magnitude_scores.append(magnitude)
            neck_force_slope_scores.append(slope)

        def valid(scores):
            return [s for s in scores if s is not None]

        valid_scores = valid(neck_force_iso_scores)
        if valid_scores:
            print_and_write(
                'NeckForce average ISO score: '
                f"{np.mean(valid_scores):.2f} "
                f"(Corridor: {np.mean(valid(neck_force_corridor_scores)):.2f}, "
                f"Phase: {np.mean(valid(neck_force_phase_scores)):.2f}, "
                f"Magnitude: {np.mean(valid(neck_force_magnitude_scores)):.2f}, "
                f"Slope: {np.mean(valid(neck_force_slope_scores)):.2f})"
            )
        else:
            print_and_write('NeckForce - no valid ISO score')

        detailed_metrics = {
            'corridor': neck_force_corridor_scores,
            'phase': neck_force_phase_scores,
            'magnitude': neck_force_magnitude_scores,
            'slope': neck_force_slope_scores,
            'overall': neck_force_iso_scores,
        }

        iso_histogram_path = plot_iso_scores_histograms(detailed_metrics, save_dir)
        print_and_write(f'ISO histogram saved to: {iso_histogram_path}')

    images_dir = os.path.join(save_dir, 'prediction_images')
    os.makedirs(images_dir, exist_ok=True)

    for i in range(min(20, len(y_test_neck_force_true))):
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.plot(y_test_neck_force_true[i], 'g-', label='True', linewidth=1, alpha=0.8)
        ax.plot(y_test_neck_force_pred[i], 'r--', label='Predict', linewidth=1, alpha=0.8)

        if i < len(neck_force_iso_scores) and neck_force_iso_scores[i] is not None:
            ax.text(
                0.02,
                0.98,
                f'ISO Score: {neck_force_iso_scores[i]:.2f}',
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            )

        ax.set_title(f'NeckForce - Sample {i + 1}')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Neck Force')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        image_path = os.path.join(images_dir, f'prediction_sample_{i + 1}.png')
        plt.savefig(image_path, dpi=300, bbox_inches='tight')
        plt.close()

    return model


if __name__ == '__main__':
    model = main()
