import numpy as np
import librosa
import os

def calculate_mfcc(audio, sr, n_mfcc=13):
    """Tính MFCC và trả về trung bình của các hệ số."""
    try:
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfcc, axis=1)  # Shape: (n_mfcc,)
    except Exception as e:
        print(f"Lỗi khi tính MFCC: {e}")
        return np.zeros(n_mfcc)

def calculate_lsfm(audio, frame_length=2048, hop_length=512):
    """Tính Long-Term Spectral Flatness Measure (LSFM), tránh lỗi inf."""
    try:
        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
        lsfm = []
        for frame in frames.T:
            fft_frame = np.abs(np.fft.fft(frame))
            if np.sum(fft_frame) < 1e-10:  # Kiểm tra nếu khung gần 0
                lsfm.append(1.0)  # Gán SFM = 1 cho nhiễu trắng hoặc im lặng
                continue
            gm = np.exp(np.mean(np.log(fft_frame + 1e-10)))  # Tránh log(0)
            am = np.mean(fft_frame)
            sfm = gm / (am + 1e-10)  # Tránh chia cho 0
            lsfm.append(np.clip(sfm, 0, 1))  # Giới hạn trong [0, 1]
        return np.mean(lsfm) if lsfm else 1.0
    except Exception as e:
        print(f"Lỗi khi tính LSFM: {e}")
        return 1.0

def calculate_zcr(audio, frame_length=2048, hop_length=512):
    """Tính Zero-Crossing Rate (ZCR)."""
    try:
        zcr = librosa.feature.zero_crossing_rate(audio, frame_length=frame_length, hop_length=hop_length)
        return np.mean(zcr)
    except Exception as e:
        print(f"Lỗi khi tính ZCR: {e}")
        return 0.0

def classify_audio(audio, sr, frame_length=2048, hop_length=512):
    """Phân loại âm thanh dựa trên MFCC, LSFM, ZCR."""
    # Kiểm tra dữ liệu đầu vào
    if np.max(np.abs(audio)) < 1e-6:
        print("Cảnh báo: Âm thanh gần như im lặng.")
        return "Noise"

    # Tính metric
    mfcc = calculate_mfcc(audio, sr)
    lsfm = calculate_lsfm(audio, frame_length, hop_length)
    zcr = calculate_zcr(audio, frame_length, hop_length)

    # In giá trị metric
    print(f"MFCC (trung bình các hệ số): \n {mfcc}")
    print(f"LSFM: {lsfm:.3f}")
    print(f"ZCR: {zcr:.3f}")

    # Phân loại dựa trên ngưỡng
    mfcc_energy = np.mean(np.abs(mfcc))  # Đơn giản hóa MFCC
    if lsfm > 0.8 and zcr > 0.3 and mfcc_energy < 50:
        return "Noise"
    elif 0.3 <= lsfm <= 0.8 and 0.1 <= zcr <= 0.3 and 50 <= mfcc_energy <= 150:
        return "Semi-Noise"
    else:
        return "None Noise"

def process_audio_folder(folder_path):
    """Xử lý tất cả file âm thanh trong thư mục."""
    # Kiểm tra thư mục
    if not os.path.exists(folder_path):
        print(f"Thư mục '{folder_path}' không tồn tại!")
        return

    # Duyệt file trong thư mục
    supported_formats = ('.wav', '.mp3', '.flac')  # Các định dạng hỗ trợ
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(supported_formats):
            file_path = os.path.join(folder_path, filename)
            print(f"\nXử lý file: {filename}")
            try:
                # Load file âm thanh
                audio, sr = librosa.load(file_path, sr=None)
                # Phân loại
                result = classify_audio(audio, sr)
                print(f"Kết quả phân loại: {result}")
            except Exception as e:
                print(f"Lỗi khi xử lý file {filename}: {e}")
        else:
            print(f"Bỏ qua file không hỗ trợ: {filename}")

# Ví dụ sử dụng
if __name__ == "__main__":
    folder_path = "audios"  # Thay bằng đường dẫn thư mục của bạn
    process_audio_folder(folder_path)