import degirum as dg
import degirum_tools
import lancedb
import numpy as np
import cv2
import uuid
import logging
from pathlib import Path
from typing import List, Any, Tuple
from lancedb.pydantic import LanceModel, Vector


class FaceRecognitionSchema(LanceModel):
    id: str
    vector: Vector(512)
    entity_name: str

    @classmethod
    def prepare_face_records(cls, face_embeddings: List[np.ndarray], entity_name: str) -> List['FaceRecognitionSchema']:
        return [
            cls(
                id=str(uuid.uuid4()),
                vector=np.array(embedding, dtype=np.float32),
                entity_name=entity_name
            )
            for embedding in face_embeddings
        ]


class FaceRecognitionSystem:
    def __init__(self, 
                 face_det_model_name: str,
                 face_rec_model_name: str,
                 inference_host_address: str = "@local",
                 face_det_zoo_url: str = "degirum/models_hailort",
                 face_rec_zoo_url: str = "degirum/models_hailort",
                 token: str = "",
                 db_uri: str = "./face_database",
                 table_name: str = "face"):

        # Setup
        self.token = token
        self.face_det_model = dg.load_model(
            model_name=face_det_model_name,
            inference_host_address=inference_host_address,
            zoo_url=face_det_zoo_url,
            token=self.token,
            overlay_color=(0, 255, 0)
        )
        self.face_rec_model = dg.load_model(
            model_name=face_rec_model_name,
            inference_host_address=inference_host_address,
            zoo_url=face_rec_zoo_url,
            token=self.token
        )
        self.db = lancedb.connect(uri=db_uri)
        self.table_name = table_name

        # Initialize or check table
        if table_name not in self.db.table_names():
            self.tbl = self.db.create_table(table_name, schema=FaceRecognitionSchema)
        else:
            self.tbl = self.db.open_table(table_name)
            schema_fields = [field.name for field in self.tbl.schema]
            if schema_fields != list(FaceRecognitionSchema.model_fields.keys()):
                raise RuntimeError(f"Table {table_name} has a different schema.")

    @staticmethod
    def align_and_crop(img: np.ndarray, landmarks: List[List[float]], image_size: int = 112) -> np.ndarray:
        _arcface_ref_kps = np.array([
            [38.2946, 51.6963], [73.5318, 51.5014],
            [56.0252, 71.7366], [41.5493, 92.3655],
            [70.7299, 92.2041]
        ], dtype=np.float32)

        assert len(landmarks) == 5

        ratio = float(image_size) / 112.0 if image_size % 112 == 0 else float(image_size) / 128.0
        diff_x = 0 if image_size % 112 == 0 else 8.0 * ratio

        dst = _arcface_ref_kps * ratio
        dst[:, 0] += diff_x

        M, inliers = cv2.estimateAffinePartial2D(np.array(landmarks), dst, ransacReprojThreshold=1000)
        assert np.all(inliers == True)

        return cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)

    def index_faces_from_directory(self, input_path: str) -> None:
        path = Path(input_path)
        image_files = [str(file) for file in path.rglob("*") if file.suffix.lower() in (".png", ".jpg", ".jpeg")]
        identities = [file.stem.split("_")[0] for file in path.rglob("*") if file.suffix.lower() in (".png", ".jpg", ".jpeg")]

        for identity, detected_faces in zip(identities, self.face_det_model.predict_batch(image_files)):
            if len(detected_faces.results) != 1:
                continue
            result = detected_faces.results[0]
            aligned_img = self.align_and_crop(detected_faces.image, [lm["landmark"] for lm in result["landmarks"]])
            face_embedding = self.face_rec_model(aligned_img).results[0]["data"][0]
            records = FaceRecognitionSchema.prepare_face_records([face_embedding], identity)
            if records:
                self.tbl.add(data=records)

    def recognize_faces(self, image_path: str, threshold: float = 0.3) -> Tuple[List[str], List[float]]:
        detected_faces = self.face_det_model(image_path)
        identities, scores = [], []

        for face in detected_faces.results:
            aligned_img = self.align_and_crop(detected_faces.image, [lm["landmark"] for lm in face["landmarks"]])
            embedding = self.face_rec_model(aligned_img).results[0]["data"][0]
            search_result = (
                self.tbl.search(embedding, vector_column_name="vector")
                .metric("cosine")
                .limit(1)
                .to_list()
            )

            if search_result:
                score = round(1 - search_result[0]["_distance"], 2)
                identity = search_result[0]["entity_name"] if score >= threshold else "Unknown"
            else:
                score = 0.0
                identity = "Unknown"

            identities.append(identity)
            scores.append(score)

        return identities, scores
