import os
import cv2
import copy
import argparse
import insightface
import onnxruntime
import numpy as np
from PIL import Image
from typing import List, Union, Dict, Set, Tuple
from pathlib import Path


class FaceSwapper:
    def __init__(self, checkpoints_dir: Path, preferred_provider: str = None):
        self.checkpoints_dir = checkpoints_dir
        self.providers = self._get_optimal_provider(preferred_provider)
        self.face_analyser = self._init_face_analyser(checkpoints_dir)
        self.face_swapper = self._init_face_swapper(checkpoints_dir)

    def _get_optimal_provider(self, preferred_provider: str = None):
        available_providers = onnxruntime.get_available_providers()
        print(f"Available providers: {available_providers}")

        provider_mapping = {
            "cuda": "CUDAExecutionProvider",
            "openvino": "OpenVINOExecutionProvider",
            "dml": "DmlExecutionProvider",
            "cpu": "CPUExecutionProvider",
        }

        if preferred_provider:
            mapped_provider = provider_mapping.get(preferred_provider.lower())
            if mapped_provider and mapped_provider in available_providers:
                print(f"Using preferred provider: {mapped_provider}")
                return [mapped_provider]

        # Default fallback logic
        if "CUDAExecutionProvider" in available_providers:
            print("Using CUDA provider")
            return ["CUDAExecutionProvider"]
        elif "DmlExecutionProvider" in available_providers:
            print("Using DirectML provider")
            return ["DmlExecutionProvider"]
        else:
            print("Using CPU provider")
            return ["CPUExecutionProvider"]

    def _init_face_analyser(self, checkpoints_dir: Path, det_size=(320, 320)):
        try:
            face_analyser = insightface.app.FaceAnalysis(
                name="buffalo_l",
                root=checkpoints_dir,
                providers=self.providers,
            )
        except Exception as e:
            print(f"Error loading face analyser: {e}")
            return None
        face_analyser.prepare(ctx_id=0, det_size=det_size)
        return face_analyser

    def _init_face_swapper(self, checkpoints_dir: Path):
        model_path = os.path.join(checkpoints_dir, "models/inswapper_128.onnx")
        try:
            return insightface.model_zoo.get_model(model_path, providers=self.providers)
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def _get_one_face(self, frame: np.ndarray):
        face = self.face_analyser.get(frame)
        try:
            return min(face, key=lambda x: x.bbox[0])
        except ValueError:
            return None

    def _get_many_faces(self, frame: np.ndarray):
        try:
            face = self.face_analyser.get(frame)
            return sorted(face, key=lambda x: x.bbox[0])
        except IndexError:
            return None

    def _swap_face(
        self, source_faces, target_faces, source_index, target_index, temp_frame
    ):
        source_face = source_faces[source_index]
        target_face = target_faces[target_index]
        return self.face_swapper.get(
            temp_frame, target_face, source_face, paste_back=True
        )

    def process(
        self,
        source_img: Union[Image.Image, List],
        target_img: Image.Image,
        source_indexes: str = "-1",
        target_indexes: str = "-1",
    ) -> Image.Image:

        target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
        target_faces = self._get_many_faces(target_img)
        num_target_faces = len(target_faces)
        num_source_images = len(source_img)

        if target_faces is None:
            print("No target faces found!")
            return target_img

        temp_frame = copy.deepcopy(target_img)

        if isinstance(source_img, list) and num_source_images == num_target_faces:
            print("Replacing faces in target image from the left to the right by order")
            for i in range(num_target_faces):
                source_faces = self._get_many_faces(
                    cv2.cvtColor(np.array(source_img[i]), cv2.COLOR_RGB2BGR)
                )
                if source_faces is None:
                    raise Exception("No source faces found!")
                temp_frame = self._swap_face(
                    source_faces, target_faces, i, i, temp_frame
                )

        elif num_source_images == 1:
            source_faces = self._get_many_faces(
                cv2.cvtColor(np.array(source_img[0]), cv2.COLOR_RGB2BGR)
            )
            if source_faces is None:
                raise Exception("No source faces found!")

            num_source_faces = len(source_faces)
            print(f"Source faces: {num_source_faces}")
            print(f"Target faces: {num_target_faces}")

            if target_indexes == "-1":
                num_iterations = min(
                    num_target_faces,
                    num_source_faces if num_source_faces > 1 else num_target_faces,
                )

                for i in range(num_iterations):
                    source_index = 0 if num_source_faces == 1 else i
                    temp_frame = self._swap_face(
                        source_faces, target_faces, source_index, i, temp_frame
                    )
            else:
                source_indexes = (
                    source_indexes.split(",")
                    if source_indexes != "-1"
                    else list(map(str, range(num_source_faces)))
                )
                target_indexes = (
                    target_indexes.split(",")
                    if target_indexes != "-1"
                    else list(map(str, range(num_target_faces)))
                )

                if len(source_indexes) != len(target_indexes):
                    raise Exception("Number of source and target indexes must match")

                for src_idx, tgt_idx in zip(source_indexes, target_indexes):
                    source_index = int(src_idx)
                    target_index = int(tgt_idx)

                    if (
                        source_index >= num_source_faces
                        or target_index >= num_target_faces
                    ):
                        raise ValueError(
                            f"Invalid index: source={source_index}, target={target_index}"
                        )

                    temp_frame = self._swap_face(
                        source_faces,
                        target_faces,
                        source_index,
                        target_index,
                        temp_frame,
                    )
        else:
            raise Exception("Unsupported face configuration")

        return Image.fromarray(cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB))

    def process_frame(
        self,
        frame: np.ndarray,
        source_img: Union[Image.Image, List],
        source_indexes: str = "-1",
    ) -> np.ndarray:
        """
        Process a single frame from webcam or video stream
        """
        try:
            target_faces = self._get_many_faces(frame)
        except Exception as e:
            raise Exception(f"Error getting target_faces: {e}")

        if target_faces is None:
            return frame
        try:
            temp_frame = copy.deepcopy(frame)
            num_target_faces = len(target_faces)
        except Exception as e:
            raise Exception(f"Error copying frame: {e}")
        try:
            source_faces = self._get_many_faces(
                cv2.cvtColor(np.array(source_img[0]), cv2.COLOR_RGB2BGR)
            )
        except Exception as e:
            raise Exception(f"Error getting source_faces: {e}")
        if source_faces is None:
            return frame

        num_source_faces = len(source_faces)
        num_iterations = min(
            num_target_faces,
            num_source_faces if num_source_faces > 1 else num_target_faces,
        )
        try:
            for i in range(num_iterations):
                source_index = 0 if num_source_faces == 1 else i
                temp_frame = self._swap_face(
                    source_faces, target_faces, source_index, i, temp_frame
                )
        except Exception as e:
            raise Exception(f"Error swapping faces: {e}")

        return temp_frame


def main():
    parser = argparse.ArgumentParser(description="Face swap.")
    parser.add_argument(
        "--source_img",
        type=str,
        required=True,
        help="The path of source image, it can be multiple images, dir;dir2;dir3.",
    )
    parser.add_argument(
        "--target_img", type=str, required=True, help="The path of target image."
    )
    parser.add_argument(
        "--output_img",
        type=str,
        default="result.png",
        help="The path and filename of output image.",
    )
    parser.add_argument(
        "--source_indexes",
        type=str,
        default="-1",
        help="Comma separated list of source face indexes",
    )
    parser.add_argument(
        "--target_indexes",
        type=str,
        default="-1",
        help="Comma separated list of target face indexes",
    )
    args = parser.parse_args()

    source_img_paths = args.source_img.split(";")
    source_img = [Image.open(img_path) for img_path in source_img_paths]
    target_img = Image.open(args.target_img)

    swapper = FaceSwapper("./checkpoints/inswapper_128.onnx")
    result_image = swapper.process(
        source_img, target_img, args.source_indexes, args.target_indexes
    )
    result_image.save(args.output_img)
    print(f"Result saved successfully: {args.output_img}")


if __name__ == "__main__":
    main()
