Project Overview

As a team, we developed an intelligent document analysis system capable of extracting and prioritizing the most relevant sections from a collection of PDF documents. The relevance is determined based on a specific user "persona" and their "job-to-be-done." Our solution is designed to operate efficiently within strict resource constraints, including offline execution and a minimal Docker image size.

Features

Robust PDF Parsing: We use PyMuPDF for efficient text and metadata extraction from PDF documents.

Advanced Heading Detection: We implement sophisticated heuristics (inherited from Round 1A) to accurately identify document titles and hierarchical headings (H1, H2, H3, H4) based on font size, boldness, positioning, and content patterns.

Semantic Search & Ranking: We employ the all-MiniLM-L6-v2 Sentence Transformer model to embed document sections and the user's query into a shared semantic space.

Maximal Marginal Relevance (MMR): We apply the MMR algorithm to rerank initial similarity results. This balances relevance to the query with diversity among the top-selected sections, ensuring a comprehensive overview.

Granular Sub-Section Analysis: We provide refined text snippets (top 3 relevant sentences) from each extracted section for deeper insights.

Offline Execution: Our solution is configured to run entirely offline once the Docker image is built, adhering to hackathon constraints.

Optimized Docker Image: We utilize a multi-stage Docker build process and targeted dependency copying to keep the final image size minimal (well under 1GB).

Constraints Compliance

CPU Only: Our solution is built and configured for CPU-only execution.

Model Size 
le 1GB: The core machine learning model (all-MiniLM-L6-v2) used is approximately 80MB, well under the 1GB limit. The final Docker image size is also significantly optimized to be under 1GB.

No Internet Access During Execution: The Docker container is run with --network none, ensuring strict offline operation.

Processing Time: We have optimized the solution for efficient processing, especially for document collections as described in the challenge.

Setup Instructions

Prerequisites

Docker Desktop (or Docker Engine) installed and running on your system.

Project Structure and File Placement

Ensure your project directory is structured as follows:

adobe-hackathon-r1b-solution/
├── input/
│   ├── YOUR_INPUT_FILE.json  <-- This file defines documents, persona, job (e.g., challenge1b_input.json).
│   └── PDFs/                <-- This directory contains all PDF files listed in the JSON.
│       ├── Learn Acrobat - Create and Convert_1.pdf
│       └── ... (all your PDF files for the challenge)
├── output/                 <-- (This directory will be created automatically, results saved here)
├── Dockerfile              <-- Docker build instructions.
├── main.py                 <-- The main Python script.
└── requirements.txt        <-- Python dependencies list.
Crucial File Placement:

YOUR_INPUT_FILE.json: Place your input JSON file (e.g., challenge1b_input.json) directly inside the input/ directory.

PDFs: All PDF files listed in your YOUR_INPUT_FILE.json must be present in the input/PDFs/ subdirectory.

Build the Docker Image

Navigate to your your-project-folder (the root of your project where Dockerfile is located) in your terminal and execute the following command:

Bash
docker build --platform linux/amd64 -t adobe-hackathon-r1b-solution .
--platform linux/amd64: Ensures the image is built for amd64 architecture, typically required for hackathon submission.

-t adobe-hackathon-r1b-solution: Tags the built image with a recognizable name.

.: Specifies that the Dockerfile is in the current directory.

Expected Behavior: The build should complete successfully without critical errors. The first build may take some time (e.g., 5-10 minutes) as it downloads models and compiles libraries. Subsequent builds will be faster due to Docker's layer caching.

Usage

Once the Docker image is successfully built, you can run the document analysis:

Navigate to your your-project-folder in your terminal and execute:

Bash
docker run --rm \
  -v "$(pwd)/input:/app/input" \
  -v "$(pwd)/output:/app/output" \
  --network none \
  adobe-hackathon-r1b-solution \
  python main.py \
  --input_json_filename "YOUR_INPUT_FILE.json" \
  --persona "HR professional" \
  --job "Create and manage fillable forms for onboarding and compliance."
--input_json_filename "YOUR_INPUT_FILE.json": Replace "YOUR_INPUT_FILE.json" with the actual name of your input JSON file (e.g., "challenge1b_input.json").

--persona "HR professional" and --job "...": These arguments specify the persona and job-to-be-done. If provided, they will override the persona and job defined in your input JSON file. If omitted, the script will read them directly from the JSON.

--network none: Crucially, this disables all network access for the container, ensuring strict offline execution.

After execution, a challenge1b_output.json file will be generated in your local output/ directory containing the analysis results.

