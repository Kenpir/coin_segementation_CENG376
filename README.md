# Coin Segmentation and Value Summation via Classical Image Processing

**Student:** Kaan Erçelik  
**Student ID:** 220218050  

## Project Overview
In this project, I developed a classical image processing pipeline that detects coins from a single tabletop image and calculates their total monetary value. The main goal is to segment each coin, estimate its real-world diameter, map it to a known denomination, and compute the total sum automatically.

The project is intentionally limited to classical image processing techniques and does not use any machine learning or deep learning methods.

## Problem Definition
Counting large numbers of coins manually is time-consuming and error-prone. This project aims to solve this problem by processing a single image containing multiple coins placed apart from each other and returning:
- The detected coins
- Their individual denominations
- The total monetary value

## Assumptions
- Coins are not touching each other.
- The image is taken from a top-down view.
- Coins are placed on a plain, A4-like, background.
- The largest coin in the image is assumed to be 1 TL, which is used as the scale reference.
- Normal lighting conditions are preferred (strong glare, motion blur and dark spots may reduce accuracy).

## Methodology
The processing pipeline follows these steps:

1. **Preprocessing**
   - Convert the input image to grayscale.
   - Apply Gaussian blur to reduce noise.

2. **Segmentation**
   - Use Otsu thresholding to separate coins from the background.
   - Invert the binary image since coins are darker than the background.
   - Apply morphological opening and closing to clean the mask.

3. **Coin Detection**
   - Extract external contours from the binary mask.
   - Filter out small contours using a minimum area threshold.
   - Estimate each coin using a minimum enclosing circle.

4. **Scale Estimation**
   - The largest detected coin is assumed to be a 1 TL coin.
   - Pixel-to-millimeter conversion is calculated using its known diameter.

5. **Denomination Matching**
   - Each coin’s measured diameter is compared with a fixed denomination table.
   - A tolerance of ±2–3% is used to account for measurement errors.

6. **Output Generation**
   - An overlay image showing detected coins and labels.
   - A CSV file containing coin position, radius, diameter, label, and value.
   - The total monetary value printed to the terminal.

## Technologies Used
- Python
- OpenCV
- NumPy

## Installation

It is recommended to run the project inside a virtual environment.
1. Create and activate a virtual environment.
Create the virtual environment:

```bash
python -m venv venv
```

Activate the virtual envoriment:
MacOS/Linux
```bash
source venv/bin/activate
```

Windows
```bash
venv\Scripts\activate
```

2. Install required dependencies.
```bash
pip install -r requirements.txt
```

## How to run
```bash
python src/coin_counter.py --input data/example_image.jpg
```

