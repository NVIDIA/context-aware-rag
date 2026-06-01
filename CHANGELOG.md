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

## Changelog
All notable changes to this project will be documented in this file.
The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

## [0.5.0] - 2025-04-09

### Added

- First release.

## [1.0.0] - 2025-10-14

### Summary

Major release featuring enhanced retrieval capabilities, expanded database support, and improved modular configuration system.

### Added

#### Enhanced retrieval methods

- Chain of Thought (CoT): Added reasoning-based retrieval for improved context understanding
- Vision Language Model (VLM): Integrated multi-modal retrieval capabilities for processing visual content
- Advanced Graph RAG (AdvGRAG): Implemented sophisticated graph traversal and retrieval algorithms for better performance

#### Expanded database support

- ArangoDB: Added support for graph-based document storage and retrieval
- Elasticsearch: Integrated full-text search capabilities with distributed search support

#### Configuration improvements

- Modular Configuration System: Redesigned configuration architecture for easier function and tool creation
- Simplified Addition Process: Streamlined workflow for adding new functions and tools to the system
- Enhanced Extensibility: Improved modularity enables faster development and deployment of new components

#### Experimental features

- MCP Tools: Experimental MCP tools to enable AI agents interacting with video.
- Structured Response: JSON Structured Response mode for responses from CA-RAG.

## [1.0.1] - 2025-10-14
### Summary
Minor release to add helm chart, documentation and config update and support llama 3.1 8B NIM

### Changes
- Added helm chart
- Documentation update
- Config update
- LLM support: LLaMa 3.1 8B

## [1.0.2] - 2026-01-30
### Summary
Latest release with bug fixes, Qwen3-VL support and documentation updates.

### Changes
- Bug fixes
- Qwen3-VL support
- Documentation updates
