# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Unit tests for the planner constants and query generator functions.
"""

from vss_ctx_rag.functions.rag.graph_rag.planner_constants import (
    get_planner_chunk_search_query,
    get_planner_entity_search_query,
    get_planner_subtitle_search_query,
)


class TestPlannerQueryGenerators:
    """Test cases for the AQL query generator functions in planner_constants."""

    def test_get_planner_chunk_search_query(self):
        """
        Verify that get_planner_chunk_search_query generates a valid AQL query
        with the correct collection name.
        """
        collection_name = "MyTestCollection"
        aql_query = get_planner_chunk_search_query(collection_name)

        # Check that the query is a non-empty string
        assert isinstance(aql_query, str)
        assert len(aql_query.strip()) > 0

        # Verify that the collection name is correctly inserted
        expected_collection_line = f"FOR doc IN {collection_name}_Chunk"
        assert expected_collection_line in aql_query

        # Verify some key AQL keywords to ensure it's a valid query structure
        assert "LET scoredDocs = (" in aql_query
        assert "COSINE_SIMILARITY(doc.embedding, @query)" in aql_query
        assert "SORT score DESC" in aql_query
        assert "LIMIT @topk_docs" in aql_query
        assert "RETURN {" in aql_query
        assert "text: text," in aql_query
        assert "score: avgScore," in aql_query
        assert "metadata: {" in aql_query

    def test_get_planner_entity_search_query(self):
        """
        Verify that get_planner_entity_search_query generates a valid AQL query
        with the correct collection name.
        """
        collection_name = "AnotherTestCollection"
        aql_query = get_planner_entity_search_query(collection_name)

        # Check that the query is a non-empty string
        assert isinstance(aql_query, str)
        assert len(aql_query.strip()) > 0

        # Verify that the collection name is correctly inserted in multiple places
        expected_entity_collection = f"FOR doc IN {collection_name}_Entity"
        expected_edge_collection = (
            f"FOR chunk IN 1..1 INBOUND entity {collection_name}_HAS_ENTITY"
        )

        assert expected_entity_collection in aql_query
        assert expected_edge_collection in aql_query

        # Verify some key AQL keywords and logic
        assert "LET scoredDocs = (" in aql_query
        assert "COSINE_SIMILARITY(doc.embedding, @query)" in aql_query
        assert "LIMIT @topk_docs" in aql_query
        assert "LET chunks = (" in aql_query
        assert "COLLECT c = chunk WITH COUNT INTO freq" in aql_query
        assert "SORT freq DESC" in aql_query
        assert "RETURN {" in aql_query
        assert "text: textformatted," in aql_query
        assert "score: avgScore," in aql_query
        assert "metadata: {" in aql_query

    def test_get_planner_subtitle_search_query(self):
        """
        Verify that get_planner_subtitle_search_query currently returns an empty string
        as it is not implemented.
        """
        collection_name = "any_collection"
        result = get_planner_subtitle_search_query(collection_name)

        # The function is expected to return an empty string
        assert result == ""
