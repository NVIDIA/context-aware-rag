<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
 *
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
 *
http://www.apache.org/licenses/LICENSE-2.0
 *
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# PDF Q&A Example

This example demonstrates how to use NVIDIA's Context-Aware RAG (CA-RAG) system for PDF document processing and question answering. The example shows how to:
- Extract text and tables from PDF documents
- Ingest documents into the CA-RAG system
- Perform question answering on the ingested documents

### Prerequisites

1. NVIDIA API Keys:
   - Get your API keys from: [build.nvidia.com](https://build.nvidia.com)
   - Export the following environment variables:
     ```bash
     export NVIDIA_BUILD_API_KEY=your_build_api_key
     export NVIDIA_API_KEY=your_api_key
     ```

2. Install NV-Ingest:
   - Follow the installation instructions at: [NV Ingest](https://github.com/NVIDIA/nv-ingest/)

### Setup

1. Start the required services using docker-compose:
   - [Docker Deployment](../guides/docker/compose.md)

2. Start the NV-Ingest client:
   - The example uses a pipeline configuration with Milvus for vector storage

### Usage

1. **Document Processing**:
   - Place your PDF documents in the `data/` directory
   - The example processes PDFs to extract:
     - Text content
     - Tables
     - Charts
     - Note: Image extraction is not supported in the current version

2. **Document Ingestion**:
   - Documents are processed and uploaded to the ingestion service
   - The system maintains document order and metadata
   - A terminating document is added to mark the end of the document set

3. **Question Answering**:
   - Initialize the retrieval service with the same UUID used for ingestion
   - Send questions to the retrieval service
   - Receive answers based on the ingested document content

### Example Notebook

The [pdf_qna.ipynb](https://github.com/NVIDIA/context-aware-rag/blob/main/examples/pdf_qna.ipynb) notebook provides a step-by-step walkthrough of the entire process, including:
- NV-Ingest Installation
- Service initialization
- Document processing and ingestion
- Question answering examples

### Notes

- The current version does not support image extraction from PDFs
- For optimal PDF processing, especially for scanned documents, consider using the `nemoretriever_parse` extraction method
- The system uses markdown format for text extraction output
