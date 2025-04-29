import cv2
import numpy as np
from picamera2 import Picamera2
from face_recognition_system import FaceRecognitionSystem
from anti_spoof.FaceAntiSpoofing import AntiSpoof
import argparse


def check_zero_to_one(value):
    fvalue = float(value)
    if fvalue <= 0 or fvalue >= 1:
        raise argparse.ArgumentTypeError("%s is an invalid value" % value)
    return fvalue


p = argparse.ArgumentParser(description="Spoofing attack detection on videostream")
p.add_argument(
    "--input", "-i", type=str, default=None, help="Path to video for predictions"
)
p.add_argument(
    "--output", "-o", type=str, default=None, help="Path to save processed video"
)
p.add_argument(
    "--model_path",
    "-m",
    type=str,
    default="anti_spoof_models/AntiSpoofing_bin_1.5_128.onnx",
    help="Path to ONNX model",
)
p.add_argument(
    "--threshold",
    "-t",
    type=check_zero_to_one,
    default=0.5,
    help="real face probability threshold above which the prediction is considered true",
)
args = p.parse_args()
anti_spoof = AntiSpoof(args.model_path)


RECOGNITION_SCORE=0.7
ANTISPOOFING_SCORE=0.0


def increased_crop(img, bbox: tuple, bbox_inc: float = 1.5):
    # Crop face based on its bounding box
    real_h, real_w = img.shape[:2]

    x, y, w, h = bbox
    w, h = w - x, h - y
    l = max(w, h)

    xc, yc = x + w / 2, y + h / 2
    x, y = int(xc - l * bbox_inc / 2), int(yc - l * bbox_inc / 2)
    x1 = 0 if x < 0 else x
    y1 = 0 if y < 0 else y
    x2 = real_w if x + l * bbox_inc > real_w else x + int(l * bbox_inc)
    y2 = real_h if y + l * bbox_inc > real_h else y + int(l * bbox_inc)

    img = img[y1:y2, x1:x2, :]
    img = cv2.copyMakeBorder(
        img,
        y1 - y,
        int(l * bbox_inc - y2 + y),
        x1 - x,
        int(l * bbox_inc) - x2 + x,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )
    return img


# 初始化辨識系統
system = FaceRecognitionSystem(
    face_det_model_name="scrfd",
    face_rec_model_name="arcface_mobilefacenet--112x112_quant_hailort_hailo8l_1",
    inference_host_address="@local",
    face_det_zoo_url="./model/scrfd.json",
    face_rec_zoo_url="./model/arcface_mobilefacenet--112x112_quant_hailort_hailo8l_1/arcface_mobilefacenet--112x112_quant_hailort_hailo8l_1.json",
    db_uri="./face_database",
    table_name="face",
)

# 開啟攝影機
picam2 = Picamera2()
picam2.start()


print("開始即時辨識，按 'n' 鍵新增新臉，'q' 鍵結束")

while True:
    frame = picam2.capture_array()

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detected_faces = system.face_det_model(rgb_frame)
    frame = rgb_frame

    face_infos = []

    for face in detected_faces.results:
        box = face["bbox"]
        x, y, w, h = map(int, box)
        landmarks = [lm["landmark"] for lm in face["landmarks"]]

        aligned_img = system.align_and_crop(detected_faces.image, landmarks)
        embedding = system.face_rec_model(aligned_img).results[0]["data"][0]

        search_result = (
            system.tbl.search(embedding, vector_column_name="vector")
            .metric("cosine")
            .limit(1)
            .to_list()
        )

        if search_result:
            score = round(1 - search_result[0]["_distance"], 2)
            identity = search_result[0]["entity_name"] if score >= RECOGNITION_SCORE else "Unknown"
        else:
            identity = "Unknown"
            score = 0.0

        face_infos.append(
            {
                "box": (x, y, w, h),
                "aligned_img": aligned_img,
                "embedding": embedding,
                "identity": identity,
                "score": score,
            }
        )

    # 畫出辨識結果
    for info in face_infos:
        x, y, w, h = info["box"]
        label = f"{info['identity']} ({info['score']:.2f})"
        color = (0, 255, 0) if info["identity"] != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (w, h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        pred = anti_spoof([increased_crop(frame, info["box"], bbox_inc=1.5)])
        cv2.putText(
            frame,
            f"{'Real' if np.argmax(pred) == 0 and pred[0][0][0] > ANTISPOOFING_SCORE else 'Fake'} ({pred[0][0][0]:.2f})",
            (x, y - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

    cv2.imshow("Face Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("n"):
        unknown_faces = [info for info in face_infos if info["identity"] == "Unknown"]

        if not unknown_faces:
            print("目前沒有 Unknown 人臉可以新增。")
            continue

        # 找最大面積的 Unknown 臉
        largest_face = max(
            unknown_faces,
            key=lambda info: (info["box"][2] - info["box"][0]) * info["box"][3]
            - info["box"][1],
        )

        # 顯示這張臉給使用者確認
        cv2.imshow("Captured Face", largest_face["aligned_img"])
        key = cv2.waitKey(0)
        print("顯示最大人臉。請確認是否要新增到資料庫。")
        print("提示：\n- 輸入名字 ➔ 新增\n- 輸入 c 或 cancel ➔ 取消\n- 按 Enter ➔ 取消")

        confirm = (
            input("請輸入新的人名（或輸入 c / cancel / 按Enter取消）：").strip().lower()
        )
        cv2.destroyWindow("Captured Face")

        if confirm in ["", "c", "cancel"]:
            print("已取消新增。")
        else:
            from face_recognition_system import FaceRecognitionSchema

            record = FaceRecognitionSchema.prepare_face_records(
                [largest_face["embedding"]], confirm
            )
            system.tbl.add(data=record)
            print(f"新增了 {confirm} 到資料庫！")


cv2.destroyAllWindows()
