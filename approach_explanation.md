Approach Explanation for Adobe "Connecting the Dots" Hackathon - Round 1B
Methodology: Persona-Driven Document Intelligence
Our Round 1B solution delivers an intelligent document analysis system capable of extracting and prioritizing relevant information from document collections based on a user's persona and specific job-to-be-done. This system integrates a robust document structure extraction pipeline with a powerful semantic understanding component, all while adhering to strict CPU-only, offline, and time/size constraints.

The methodology is structured around two primary, interconnected phases:

Phase 1: Advanced Document Outline Extraction (Leveraging Round 1A Foundation)

This initial phase is a direct integration of our highly refined heuristic-based solution developed for Round 1A. Its purpose is to accurately parse each PDF in the input collection and extract its structural outline.

Granular Text Extraction: PyMuPDF is used to extract detailed text block information (content, font properties, bounding boxes).

Sophisticated Noise Filtering: Aggressive heuristics identify and remove irrelevant elements like page numbers, headers/footers, and TOC dot leaders, ensuring a clean content stream.

Adaptive Structural Analysis: Document-wide font statistics (average body font size, max font size) are computed to create adaptive thresholds for heading detection.

Multi-Criteria Heading Detection: Headings (H1, H2, H3, H4) are identified based on a scoring system considering font prominence (size, boldness), vertical/horizontal spacing, and powerful regular expression patterns for numbered/lettered headings and common unnumbered phrases.

Refined Title Detection & Hierarchical Validation: Flexible logic extracts the main document title (handling multi-line titles and empty titles as per samples). Post-processing ensures a logical hierarchy, merging multi-line headings and preventing illogical level jumps.

This phase is entirely CPU-based, offline, and contributes effectively 0MB to the model size, ensuring compliance. Its language-agnostic nature inherently supports multilingual documents.

Phase 2: Persona-Driven Semantic Ranking

This phase takes the structured sections and sub-sections from Phase 1 and ranks them based on their relevance to the user's query.

Semantic Embedding Generation: We utilize a pre-trained SentenceTransformer model, specifically all-MiniLM-L6-v2. This model is chosen for its excellent balance of performance, semantic accuracy, and compact size (approx. 80MB, well within the 1GB limit for Round 1B), and its ability to generate high-quality semantic embeddings. The model is downloaded during the Docker build process, guaranteeing offline execution.

Query Formulation: The persona and job-to-be-done inputs are combined into a single, comprehensive query string.

Content Segmentation: Each identified section and sub-section from all input documents is treated as a distinct content unit. The full text content for each of these units is extracted by precisely determining its boundaries (from its heading to the start of the next heading or end of the document).

Relevance Scoring: Both the formulated query and each content unit are transformed into dense vector embeddings using the SentenceTransformer. The semantic similarity between the query embedding and each content unit embedding is then calculated using cosine similarity.

Prioritization and Output Generation: All content units are ranked in descending order based on their similarity scores. The top 5 most relevant sections are then formatted into the challenge1b_output.json schema, including metadata, extracted_sections with importance_rank, and subsection_analysis with the refined_text of the relevant content.

Compliance and Performance:

The entire solution runs on CPU-only resources and is entirely offline. The all-MiniLM-L6-v2 model's size is well within the 1GB constraint. The pipeline is optimized for efficiency, aiming to process document collections (3-5 documents) within the 60-second time limit. This comprehensive, hybrid methodology ensures both high accuracy in structural extraction and precise, context-aware relevance ranking.





