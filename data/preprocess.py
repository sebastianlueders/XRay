import os
import numpy as np
import pandas as pd
import cv2


class FERPlusPreprocessor:

    def __init__(self, data_path, output_path, image_size=48):
        self.data_path = data_path
        self.output_path = output_path
        self.image_size = image_size
        self.emotion_labels = [
            'neutral', 'happiness', 'surprise', 'sadness',
            'anger', 'disgust', 'fear', 'contempt'
        ]

        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(output_path, split), exist_ok=True)

    def process(self):
        self._process_split('fer2013train.csv', 'train')
        self._process_split('fer2013valid.csv', 'val')
        self._process_split('fer2013test.csv', 'test')
        self._create_dataset_info()
        print("FERPlus preprocessing complete.")

    def _process_split(self, csv_file, split):
        df = pd.read_csv(os.path.join(self.data_path, csv_file))
        split_dir = os.path.join(self.output_path, split)
        metadata = []

        for i, row in df.iterrows():
            # convert pixel string to image
            pixels = np.array(row['pixels'].split(), dtype=np.uint8)
            image = pixels.reshape((self.image_size, self.image_size))

            # get dominant emotion
            emotion_votes = row[self.emotion_labels].values
            emotion_idx = np.argmax(emotion_votes)
            emotion_name = self.emotion_labels[emotion_idx]

            # save image
            filename = f"{split}_{i:05d}_{emotion_name}.png"
            filepath = os.path.join(split_dir, filename)
            cv2.imwrite(filepath, image)

            metadata.append({
                'filename': filename,
                'split': split,
                'emotion_idx': emotion_idx,
                'emotion': emotion_name
            })

        pd.DataFrame(metadata).to_csv(os.path.join(self.output_path, f"{split}_metadata.csv"), index=False)

    def _create_dataset_info(self):
        def count_rows(csv_name):
            path = os.path.join(self.output_path, csv_name)
            return len(pd.read_csv(path)) if os.path.exists(path) else 0

        summary = {
            'dataset_name': 'FERPlus',
            'image_size': self.image_size,
            'num_classes': len(self.emotion_labels),
            'emotion_labels': self.emotion_labels,
            'num_train': count_rows('train_metadata.csv'),
            'num_val': count_rows('val_metadata.csv'),
            'num_test': count_rows('test_metadata.csv')
        }

        pd.DataFrame([summary]).to_json(
            os.path.join(self.output_path, 'dataset_info.json'),
            orient='records',
            indent=2
        )
