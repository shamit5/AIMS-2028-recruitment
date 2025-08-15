Scene Localization in Dense Images via Natural Language Queries

This repository contains a YOLO + CLIP-based pipeline for localizing specific interactions in dense images using natural language queries. Given an image with multiple activities, the model returns the most relevant cropped region corresponding to the input text query.

📂 Repository Structure
scene-localization/
│
├── Untitled2.ipynb          # Main Colab notebook with YOLO + CLIP pipeline
├── README.md                # This file
└── requirements.txt         # Optional: dependencies if using locally

⚙️ Features

YOLOv8 for fast candidate object detection.

OpenAI CLIP (ViT-B/32) for ranking detected regions against a text query.

Shows best matching crop and optionally top-K ranked crops.

Fully compatible with Google Colab GPU runtime.

🛠 Setup Instructions

Open the notebook in Colab:

Open in Colab

Install dependencies:

!pip install -q ultralytics ftfy regex tqdm
!pip install -q git+https://github.com/openai/CLIP.git
!pip install -q pillow matplotlib opencv-python


Restart the runtime if prompted.

🚀 How to Run

Upload an image:

from google.colab import files
uploaded = files.upload()
image_path = list(uploaded.keys())[0]


Set your text query:

query = "a person selling vegetables"


Run the notebook cells:

YOLO predicts candidate bounding boxes.

CLIP re-ranks the candidates based on similarity to the query.

The notebook displays:

Original image with YOLO boxes.

Best crop according to CLIP.

Optional top-K ranked crops.

🖼 Example Workflow

Input Image

An image of a busy street market.

YOLO Candidate Boxes

All potential objects detected by YOLO.

CLIP Top-1 Crop

The crop most semantically matching the query: "a person selling vegetables".

CLIP Top-3 Crops (Optional)

Other high-ranking crops for additional context.

⚙️ Key Dependencies

Ultralytics YOLOv8

OpenAI CLIP

Python libraries: torch, opencv-python, Pillow, matplotlib, tqdm, ftfy, regex

💡 Notes

GPU runtime is recommended in Colab.

You can change the YOLO model to larger variants (yolov8s.pt, yolov8m.pt) for better detection accuracy.

Adjust conf and iou thresholds in YOLO inference to control candidate box selection.

🙏 Acknowledgements

AIMS 2K28 Recruitments for giving the opportunity to work on this deep learning project.

Open-source libraries: YOLOv8 and CLIP.

Google Colab for free GPU resources.
