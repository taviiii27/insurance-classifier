This project implements a text-based classifier that categorizes companies according to an insurance taxonomy. Using company descriptions, business tags, sector information, and other metadata, it processes and combines these text features into a unified representation.

The core classification method applies TF-IDF vectorization to convert textual data into numerical vectors, then calculates cosine similarity between company profiles and insurance taxonomy labels. If the similarity score exceeds a predefined threshold, the corresponding taxonomy label(s) are assigned to the company. Companies with no sufficiently similar labels are marked as "unclassified."

This approach provides a lightweight, interpretable way to classify companies without complex machine learning models, making it easy to customize and extend. The output is an annotated CSV file listing each company alongside its predicted insurance category labels.
