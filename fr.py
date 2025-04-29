import cv2
import numpy as np
from picamera2 import Picamera2
from face_recognition.face_recognition_system import FaceRecognitionSystem
from anti_spoof.face_anti_spoofing import AntiSpoof
from utils.image_utils import increased_crop
from dotenv import load_dotenv


load_dotenv()

# 讀取參數
MODEL_PATH = os.getenv("MODEL_PATH", "anti_spoof_models/AntiSpoofing_bin_1.5_128.onnx")
RECOGNITION_SCORE = float(os.getenv("RECOGNITION_SCORE", 0.7))
ANTISPOOFING_SCORE = float(os.getenv("ANTISPOOFING_SCORE", 0.0))

FACE_DET_MODEL_NAME = os.getenv("FACE_DET_MODEL_NAME", "scrfd")
FACE_REC_MODEL_NAME = os.getenv("FACE_REC_MODEL_NAME", "arcface_mobilefacenet--112x112_quant_hailort_hailo8l_1")
INFERENCE_HOST_ADDRESS = os.getenv("INFERENCE_HOST_ADDRESS", "@local")

FACE_DET_ZOO_URL = os.getenv("FACE_DET_ZOO_URL", "./model/scfrd_10g/scrfd.json")
FACE_REC_ZOO_URL = os.getenv("FACE_REC_ZOO_URL", "./model/arcface_mobilefacenet--112x112_quant_hailort_hailo8l_1/arcface_mobilefacenet--112x112_quant_hailort_hailo8l_1.json")

DB_URI = os.getenv("DB_URI", "./face_database")
TABLE_NAME = os.getenv("TABLE_NAME", "face")

# 初始化 AntiSpoof 模型
anti_spoof = AntiSpoof(MODEL_PATH)


# 初始化辨識系統
system = FaceRecognitionSystem(
    face_det_model_name=FACE_DET_MODEL_NAME,
    face_rec_model_name=FACE_REC_MODEL_NAME,
    inference_host_address=INFERENCE_HOST_ADDRESS,
    face_det_zoo_url=FACE_DET_ZOO_URL,
    face_rec_zoo_url=FACE_REC_ZOO_URL,
    db_uri=DB_URI,
    table_name=TABLE_NAME,
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
