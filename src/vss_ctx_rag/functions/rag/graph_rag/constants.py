# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


"""constants.py: File contains constants for graph rag"""

from vss_ctx_rag.functions.rag.graph_rag.planner_constants import (
    get_planner_chunk_search_query,
    get_planner_entity_search_query,
    get_planner_subtitle_search_query,
)
from vss_ctx_rag.utils.ctx_rag_logger import logger

### GRAPH EXTRACTION
CHUNK_VECTOR_INDEX_NAME = "vector"
DROP_CHUNK_VECTOR_INDEX_QUERY = f"DROP INDEX {CHUNK_VECTOR_INDEX_NAME} IF EXISTS;"
CREATE_CHUNK_VECTOR_INDEX_QUERY = """
CREATE VECTOR INDEX {index_name} IF NOT EXISTS FOR (c:Chunk) ON c.embedding
"""
DROP_INDEX_QUERY = "DROP INDEX {index_name} IF EXISTS;"
LABELS_QUERY = "CALL db.labels()"
FULL_TEXT_QUERY = "CREATE FULLTEXT INDEX {index_name} IF NOT EXISTS FOR (n{labels_str}) ON EACH [n.name, n.description];"
SUBTITLE_SEARCH_FULL_TEXT_QUERY = "CREATE FULLTEXT INDEX {index_name} IF NOT EXISTS FOR (n:Subtitle) ON EACH [n.text];"
FILTER_LABELS = ["Chunk", "Document", "Summary", "Subtitle", "__Community__"]

KEYWORD_SEARCH_INDEX_DROP_QUERY = "DROP INDEX {index_name} IF EXISTS;"
KEYWORD_SEARCH_FULL_TEXT_QUERY = (
    "CREATE FULLTEXT INDEX {index_name} IF NOT EXISTS FOR (n:Chunk) ON EACH [n.text]"
)

DUPLICATE_SCORE_VALUE = 0.9

### Vector graph search
VECTOR_GRAPH_SEARCH_ENTITY_LIMIT = 40
VECTOR_GRAPH_SEARCH_EMBEDDING_MIN_MATCH = 0.3
ARANGO_VECTOR_GRAPH_SEARCH_EMBEDDING_MIN_MATCH = 0.1
ARANGO_VECTOR_GRAPH_SEARCH_EMBEDDING_MAX_MATCH = 0.7
VECTOR_GRAPH_SEARCH_EMBEDDING_MAX_MATCH = 0.9
VECTOR_GRAPH_SEARCH_ENTITY_LIMIT_MINMAX_CASE = 20
VECTOR_GRAPH_SEARCH_ENTITY_LIMIT_MAX_CASE = 40

VECTOR_GRAPH_SEARCH_QUERY_PREFIX = """
        WITH node as chunk, score
        // find the document of the chunk
        MATCH (chunk)-[:PART_OF]->(d:Document)
        WHERE CASE
            WHEN $uuid IS NOT NULL THEN d.uuid = $uuid
            ELSE true
        END

        // aggregate chunk-details
        WITH d, collect(DISTINCT {chunk: chunk, score: score}) AS chunks, avg(score) as avg_score

        // fetch entities
        CALL (chunks) { WITH chunks
        UNWIND chunks as chunkScore
        WITH chunkScore.chunk as chunk
    """

VECTOR_GRAPH_SEARCH_ENTITY_QUERY = """
    OPTIONAL MATCH (chunk)-[:HAS_ENTITY]->(e)
    WITH e, count(*) AS numChunks
    ORDER BY numChunks DESC
    LIMIT {no_of_entites}

    WITH
    CASE
        WHEN e.embedding IS NULL OR ({embedding_match_min} <= vector.similarity.cosine($embedding, e.embedding) AND vector.similarity.cosine($embedding, e.embedding) <= {embedding_match_max}) THEN
            collect {{
                OPTIONAL MATCH path=(e)(()-[rels:!HAS_ENTITY&!PART_OF]-()){{0,1}}(:!Chunk&!Document)
                RETURN path LIMIT {entity_limit_minmax_case}
            }}
        WHEN e.embedding IS NOT NULL AND vector.similarity.cosine($embedding, e.embedding) >  {embedding_match_max} THEN
            collect {{
                OPTIONAL MATCH path=(e)(()-[rels:!HAS_ENTITY&!PART_OF]-()){{0,2}}(:!Chunk&!Document)
                RETURN path LIMIT {entity_limit_max_case}
            }}
        ELSE
            collect {{
                MATCH path=(e)
                RETURN path
            }}
    END AS paths, e
"""

VECTOR_GRAPH_SEARCH_QUERY_SUFFIX = """
        WITH apoc.coll.toSet(apoc.coll.flatten(collect(DISTINCT paths))) AS paths,
            collect(DISTINCT e) AS entities

        // De-duplicate nodes and relationships across chunks
        RETURN
            collect {
                UNWIND paths AS p
                UNWIND relationships(p) AS r
                RETURN DISTINCT r
            } AS rels,
            collect {
                UNWIND paths AS p
                UNWIND nodes(p) AS n
                RETURN DISTINCT n
            } AS nodes,
            entities
        }

    // Generate metadata and text components for chunks, nodes, and relationships
    WITH d, avg_score,
        [c IN chunks | c.chunk.text] AS texts,
        [c IN chunks | c.chunk.asset_dir] AS asset_dirs,
        [n IN nodes | elementId(n)] AS entityIds,
        [r IN rels | elementId(r)] AS relIds,
     apoc.coll.sort([
         n IN nodes |
         coalesce(apoc.coll.removeAll(labels(n), ['__Entity__'])[0], "") + ":" +
         coalesce(
            n.name,
            n[head([k IN keys(n) WHERE k =~ "(?i)(name|title|id|description)$"])],
            ""
        ) +
         (CASE WHEN n.description IS NOT NULL THEN " (" + n.description + ")" ELSE "" END)
     ]) AS nodeTexts,
     apoc.coll.sort([
         r IN rels |
         coalesce(apoc.coll.removeAll(labels(startNode(r)), ['__Entity__'])[0], "") + ":" +
         coalesce(
            startNode(r).name,
            startNode(r)[head([k IN keys(startNode(r)) WHERE k =~ "(?i)(name|title|id|description)$"])],
            ""
        ) + " " + type(r) + " " +
         coalesce(apoc.coll.removeAll(labels(endNode(r)), ['__Entity__'])[0], "") + ":" +
         coalesce(
            endNode(r).name,
            endNode(r)[head([k IN keys(endNode(r)) WHERE k =~ "(?i)(name|title|id|description)$"])],
            ""
        )
     ]) AS relTexts,
     entities

        // Combine texts into response text
        WITH d, avg_score, entityIds, relIds, asset_dirs,
            "Text Content:\n" + apoc.text.join(texts, "\n----\n") +
            "\n----\nEntities:\n" + apoc.text.join(nodeTexts, "\n") +
            "\n----\nRelationships:\n" + apoc.text.join(relTexts, "\n") AS text,
            entities

        RETURN
            text,
            avg_score AS score,
            {
                length: size(text),
                source: d.uuid,
                asset_dirs: asset_dirs,
                entities : {
                    entityids: entityIds,
                    relationshipids: relIds
                }
            } AS metadata
        """

GNN_VECTOR_GRAPH_SEARCH_QUERY_SUFFIX = """
        WITH apoc.coll.toSet(apoc.coll.flatten(collect(DISTINCT paths))) AS paths,
            collect(DISTINCT e) AS entities

        // De-duplicate nodes and relationships across chunks
        RETURN
            collect {
                UNWIND paths AS p
                UNWIND relationships(p) AS r
                RETURN DISTINCT r
            } AS rels,
            collect {
                UNWIND paths AS p
                UNWIND nodes(p) AS n
                RETURN DISTINCT n
            } AS nodes,
            entities
        }

    // Generate metadata and text components for chunks, nodes, and relationships
    WITH d, avg_score,
        [c IN chunks | c.chunk.text] AS texts,
        [r IN rels |
            coalesce(apoc.coll.removeAll(labels(startNode(r)), ['__Entity__'])[0], "") + ":" +
         coalesce(
            startNode(r).name,
            startNode(r)[head([k IN keys(startNode(r)) WHERE k =~ "(?i)(name|title|id|description)$"])],""
         )
        ] AS sourceNode,
        [r IN rels |
            type(r)
        ] AS edge,
        [r IN rels |
            coalesce(apoc.coll.removeAll(labels(endNode(r)), ['__Entity__'])[0], "") + ":" +
         coalesce(
            endNode(r).name,
            endNode(r)[head([k IN keys(endNode(r)) WHERE k =~ "(?i)(name|title|id|description)$"])],""
         )
        ] AS targetNode

        RETURN
            apoc.text.join(texts, "\n----\n") AS text,
            avg_score AS score,
            {
                sourceNode: sourceNode,
                edge: edge,
                targetNode: targetNode,
                chunkdetails : texts
            } AS metadata
        """

VECTOR_GRAPH_SEARCH_QUERY = (
    VECTOR_GRAPH_SEARCH_QUERY_PREFIX
    + VECTOR_GRAPH_SEARCH_ENTITY_QUERY.format(
        no_of_entites=VECTOR_GRAPH_SEARCH_ENTITY_LIMIT,
        embedding_match_min=VECTOR_GRAPH_SEARCH_EMBEDDING_MIN_MATCH,
        embedding_match_max=VECTOR_GRAPH_SEARCH_EMBEDDING_MAX_MATCH,
        entity_limit_minmax_case=VECTOR_GRAPH_SEARCH_ENTITY_LIMIT_MINMAX_CASE,
        entity_limit_max_case=VECTOR_GRAPH_SEARCH_ENTITY_LIMIT_MAX_CASE,
    )
    + VECTOR_GRAPH_SEARCH_QUERY_SUFFIX
)

GNN_VECTOR_GRAPH_SEARCH_QUERY = (
    VECTOR_GRAPH_SEARCH_QUERY_PREFIX
    + VECTOR_GRAPH_SEARCH_ENTITY_QUERY.format(
        no_of_entites=VECTOR_GRAPH_SEARCH_ENTITY_LIMIT,
        embedding_match_min=VECTOR_GRAPH_SEARCH_EMBEDDING_MIN_MATCH,
        embedding_match_max=VECTOR_GRAPH_SEARCH_EMBEDDING_MAX_MATCH,
        entity_limit_minmax_case=VECTOR_GRAPH_SEARCH_ENTITY_LIMIT_MINMAX_CASE,
        entity_limit_max_case=VECTOR_GRAPH_SEARCH_ENTITY_LIMIT_MAX_CASE,
    )
    + GNN_VECTOR_GRAPH_SEARCH_QUERY_SUFFIX
)

### Entity search
ENTITY_SEARCH_TOP_K = 10
ENTITY_SEARCH_TOP_CHUNKS = 3
ENTITY_SEARCH_TOP_OUTSIDE_RELS = 10

ENTITY_SEARCH_QUERY = """
        WITH collect(node) AS nodes,
            avg(score) AS score,
            collect({{entityids: elementId(node), score: score}}) AS metadata

        WITH score, nodes, metadata,

            collect {{
                UNWIND nodes AS n
                MATCH (n)<-[:HAS_ENTITY]->(c:Chunk)
                WITH c, count(distinct n) AS freq
                RETURN c
                ORDER BY freq DESC
                LIMIT {topChunks}
            }} AS chunks,

            collect {{
                UNWIND nodes AS n
                UNWIND nodes AS m
                MATCH (n)-[r]->(m)
                RETURN DISTINCT r
                // TODO: need to add limit
            }} AS rels,

            collect {{
                UNWIND nodes AS n
                MATCH path = (n)-[r]-(m:__Entity__)
                WHERE NOT m IN nodes
                WITH m, collect(distinct r) AS rels, count(*) AS freq
                ORDER BY freq DESC
                LIMIT {topOutsideRels}
                WITH collect(m) AS outsideNodes, apoc.coll.flatten(collect(rels)) AS rels
                RETURN {{ nodes: outsideNodes, rels: rels }}
            }} AS outside
        """

ENTITY_SEARCH_QUERY_SUFFIX = """
            RETURN {
            chunks: [c IN chunks | c.text],
            entities: [
                n IN nodes |
                CASE
                WHEN size(labels(n)) > 1 THEN
                    apoc.coll.removeAll(labels(n), ["__Entity__"])[0] + ":" + n.name + " " + coalesce(n.description, "")
                ELSE
                    n.name + " " + coalesce(n.description, "")
                END
            ],
            relationships: [
                r IN rels |
                startNode(r).name + " " + type(r) + " " + endNode(r).name
            ],
            outside: {
                nodes: [
                n IN outside[0].nodes |
                CASE
                    WHEN size(labels(n)) > 1 THEN
                    apoc.coll.removeAll(labels(n), ["__Entity__"])[0] + ":" + n.name + " " + coalesce(n.description, "")
                    ELSE
                    n.name + " " + coalesce(n.description, "")
                END
                ],
                relationships: [
                r IN outside[0].rels |
                CASE
                    WHEN size(labels(startNode(r))) > 1 THEN
                    apoc.coll.removeAll(labels(startNode(r)), ["__Entity__"])[0] + ":" + startNode(r).name + " "
                    ELSE
                    startNode(r).name + " "
                END +
                type(r) + " " +
                CASE
                    WHEN size(labels(endNode(r))) > 1 THEN
                    apoc.coll.removeAll(labels(endNode(r)), ["__Entity__"])[0] + ":" + endNode(r).name
                    ELSE
                    endNode(r).name
                END
                ]
            }
            } AS text,
            score,
            {entities: metadata} AS metadata
        """

ENTITY_SEARCH_QUERY_FORMATTED = (
    ENTITY_SEARCH_QUERY.format(
        topChunks=ENTITY_SEARCH_TOP_CHUNKS,
        topOutsideRels=ENTITY_SEARCH_TOP_OUTSIDE_RELS,
    )
    + ENTITY_SEARCH_QUERY_SUFFIX
)


### CHAT TEMPLATES
CHAT_SYSTEM_TEMPLATE = """

You are an AI-powered question-answering agent watching a video. The video summary is given below.
Your task is to provide accurate and comprehensive responses to user queries based on the video, chat history, and available resources.
Answer the questions from the point of view of someone watching the video.

### Response Guidelines:
1. **Direct Answers**: Provide clear and thorough answers to the user's queries without headers unless requested. Avoid speculative responses.
2. **Utilize History and Context**: Leverage relevant information from previous interactions, the current user input, and the context.
3. **No Greetings in Follow-ups**: Start with a greeting in initial interactions. Avoid greetings in subsequent responses unless there's a significant break or the chat restarts.
4. **Admit Unknowns**: Clearly state if an answer is unknown. Avoid making unsupported statements.
5. **Avoid Hallucination**: Only provide information relevant to the context. Do not invent information.
6. **Response Length**: Keep responses concise and relevant. Aim for clarity and completeness within 4-5 sentences unless more detail is requested.
7. **Tone and Style**: Maintain a professional and informative tone. Be friendly and approachable.
8. **Error Handling**: If a query is ambiguous or unclear, ask for clarification rather than providing a potentially incorrect answer.
9. **Summary Availability**: If the video summary is empty, do not provide answers based solely on internal knowledge. Instead, respond appropriately by indicating the lack of information.
10. **Absence of Objects**: If a query asks about objects which are not present in the video, provide an answer stating the absence of the objects in the video. Avoid giving any further explanation. Example: "No, there are no mangoes on the tree."
11. **Absence of Events**: If a query asks about an event which did not occur in the video , provide an answer which states that the event did not occur. Avoid giving any further explanation. Example: "No, the pedestrian did not cross the street."
12. **Object counting**: If a query asks the count of objects belonging to a category, only provide the count. Do not enumerate the objects.

### Example Responses:
User: Hi
AI Response: 'Hello there! How can I assist you today?'

User: "What is Langchain?"
AI Response: "Langchain is a framework that enables the development of applications powered by large language models, such as chatbots. It simplifies the integration of language models into various applications by providing useful tools and components."

User: "Can you explain how to use memory management in Langchain?"
AI Response: "Langchain's memory management involves utilizing built-in mechanisms to manage conversational context effectively. It ensures that the conversation remains coherent and relevant by maintaining the history of interactions and using it to inform responses."

User: "I need help with PyCaret's classification model."
AI Response: "PyCaret simplifies the process of building and deploying machine learning models. For classification tasks, you can use PyCaret's setup function to prepare your data. After setup, you can compare multiple models to find the best one, and then fine-tune it for better performance."

User: "What can you tell me about the latest realtime trends in AI?"
AI Response: "I don't have that information right now. Is there something else I can help with?"

### Video Summary:
<summary>
{context}
</summary>


**IMPORTANT** : YOUR KNOWLEDGE FOR ANSWERING THE USER'S QUESTIONS IS LIMITED TO THE SUMMARY OF A VIDEO GIVEN ABOVE.


Note: This system does not generate answers based solely on internal knowledge. It answers from the information provided in the user's current and previous inputs, and from the context.
"""

QUESTION_TRANSFORM_TEMPLATE = """
Given the below conversation, generate a search query to look up in order to get information relevant to the conversation.
Only provide information relevant to the context. Do not invent information.
Only respond with the query, nothing else.
"""

## CHAT QUERIES
VECTOR_SEARCH_TOP_K = 5

## CHAT SETUP
CHAT_SEARCH_KWARG_SCORE_THRESHOLD = 0.5
CHAT_EMBEDDING_FILTER_SCORE_THRESHOLD = 0.10

QUERY_TO_DELETE_UUID_GRAPH = """
MATCH (d:Document {uuid:$uuid})
OPTIONAL MATCH (d)<-[:PART_OF]-(c:Chunk) // Get all chunks for this document

// Collect all chunks belonging to this document
WITH d, collect(c) as docChunks

// Find entities linked ONLY to these chunks
OPTIONAL MATCH (e)<-[:HAS_ENTITY]-(entityChunk:Chunk)
WHERE entityChunk in docChunks // Entity linked to one of our chunks
WITH d, docChunks, e, entityChunk
WHERE e IS NOT NULL AND NOT EXISTS { // Ensure entity is not linked elsewhere
    MATCH (e)<-[:HAS_ENTITY]-(otherChunk:Chunk)
    WHERE NOT otherChunk IN docChunks
}
// Collect unique orphaned entities
WITH d, docChunks, collect(DISTINCT e) as orphanedEntities

// Find summaries linked ONLY to these chunks
OPTIONAL MATCH (s:Summary)<-[:IN_SUMMARY]-(summaryChunk:Chunk)
WHERE summaryChunk in docChunks // Summary linked to one of our chunks
WITH d, docChunks, orphanedEntities, s, summaryChunk
WHERE s IS NOT NULL AND NOT EXISTS { // Ensure summary is not linked elsewhere
    MATCH (s)<-[:IN_SUMMARY]-(otherChunk:Chunk)
    WHERE NOT otherChunk IN docChunks
}
// Collect unique orphaned summaries
WITH d, docChunks, orphanedEntities, collect(DISTINCT s) as orphanedSummaries

// Now, detach and delete the document, its chunks, and the orphaned entities/summaries
// Ensure we only add non-null nodes to the list to delete
WITH [n IN docChunks WHERE n IS NOT NULL] +
     [n IN orphanedEntities WHERE n IS NOT NULL] +
     [n IN orphanedSummaries WHERE n IS NOT NULL] +
     [d] as nodesToDelete
UNWIND nodesToDelete as nodeToDelete
DETACH DELETE nodeToDelete
"""


def get_chunk_search_query(collection_name: str):
    """
    Get the AQL query for vector graph search

    Args:
        params (dict, optional): Optional parameters to customize the query

    Returns:
        str: The AQL query string
    """

    aql = f"""
        // Vector search step to find relevant chunks
        LET scoredDocs = (
            FOR doc IN {collection_name}_Chunk
                LET score = COSINE_SIMILARITY(doc.embedding, @query)
                SORT score DESC
                LIMIT @topk_docs
                RETURN {{node: doc, score: score}}
        )

        // Calculate average score directly from scoredDocs
        LET avgScore = AVERAGE(scoredDocs[*].score)

        // Process the chunks
        LET chunks = scoredDocs

        // Fetch entities connected to chunks
        LET chunkEntities = (
            FOR chunkData IN chunks
                LET chunk = chunkData.node

                // For each chunk, find connected entities via HAS_ENTITY edges
                FOR entity IN 1..1 OUTBOUND chunk {collection_name}_HAS_ENTITY
                    FILTER entity.label == "__Entity__"

                    // Group and count entities across chunks
                    COLLECT e = entity WITH COUNT INTO numChunks
                    SORT numChunks DESC
                    LIMIT {VECTOR_GRAPH_SEARCH_ENTITY_LIMIT}

                    LET sim = e.embedding == null ? null  : COSINE_SIMILARITY(e.embedding, @query)

                    LET paths = (
                        RETURN (
                            (sim == null OR ({ARANGO_VECTOR_GRAPH_SEARCH_EMBEDDING_MIN_MATCH} <= sim AND sim <= {ARANGO_VECTOR_GRAPH_SEARCH_EMBEDDING_MAX_MATCH}))
                            ? (
                                // Traverse with depth 1
                                FOR v,edge,p IN 1..1 ANY e {collection_name}_LINKS_TO
                                LIMIT {VECTOR_GRAPH_SEARCH_ENTITY_LIMIT_MINMAX_CASE}
                                RETURN p
                            )
                            : (
                                (sim != null AND sim > {ARANGO_VECTOR_GRAPH_SEARCH_EMBEDDING_MAX_MATCH})
                                ? (
                                    // Traverse with depth 2
                                    FOR v,edge,p IN 1..2 ANY e {collection_name}_LINKS_TO
                                    LIMIT {VECTOR_GRAPH_SEARCH_ENTITY_LIMIT_MAX_CASE}
                                    RETURN p
                                )
                                : (
                                    // Fallback case
                                    [ {{ vertices: [ e ], edges: [] }} ]
                                )
                            )
                        )
                    )
                    RETURN {{ e, paths }}
            )

        // Collect all unique nodes and edges from paths
        LET allPaths = FLATTEN(
            FOR pe IN chunkEntities
                RETURN pe.paths
        )

        LET allEntities = (
            FOR pe IN chunkEntities
                RETURN pe.entity
        )

        // De-duplicate nodes and relationships across chunks
        LET uniqueRelationships = UNIQUE(
            FLATTEN(
                FOR p IN FLATTEN(allPaths)
                    FILTER p.edges != null
                    RETURN p.edges
            )
        )

        LET uniqueNodes = UNIQUE(
            FLATTEN(
                FOR p IN FLATTEN(allPaths)
                    FILTER p.vertices != null
                    RETURN p.vertices
            )
        )

        // Generate texts from chunks
        LET texts = (
            FOR c IN chunks
                RETURN c.node.text
        )

        // Generate node texts - equivalent to the Cypher nodeTexts logic
        LET nodeTexts = (
            FOR n IN uniqueNodes
                LET labels = n.type ? SPLIT(n.type, ":") : []
                LET mainLabel = LENGTH(labels) > 0 &&
                                labels[0] != "__Entity__" ?
                                labels[0] : ""
                LET description = n.description != null ?
                                CONCAT(" (", n.description, ")") : ""

                RETURN CONCAT(mainLabel, ":", n.name, description)
        )

        // Generate relationship texts - equivalent to the Cypher relTexts logic
        LET relTexts = (
            FOR r IN uniqueRelationships
                LET startNode = DOCUMENT(r._from)
                LET endNode = DOCUMENT(r._to)

                LET startLabels = startNode.label ? SPLIT(startNode.label, ":") : []
                LET startLabel = LENGTH(startLabels) > 0 &&
                                startLabels[0] != "__Entity__" ?
                                startLabels[0] : ""

                LET endLabels = endNode.label ? SPLIT(endNode.label, ":") : []
                LET endLabel = LENGTH(endLabels) > 0 &&
                            endLabels[0] != "__Entity__" ?
                            endLabels[0] : ""

                RETURN CONCAT(
                    startLabel, ":", startNode.name, " ",
                    r.type, " ",
                    endLabel, ":", endNode.name
                )
        )

        // Combine texts into response text - equivalent to the Cypher text formatting
        LET responseText = CONCAT(
            "Text Content:\n",
            CONCAT_SEPARATOR("\n----\n", texts),
            "\n----\nEntities:\n",
            CONCAT_SEPARATOR("\n", nodeTexts),
            "\n----\nRelationships:\n",
            CONCAT_SEPARATOR("\n", relTexts)
        )

        // Collect asset_dirs from chunks
        LET asset_dirs = (
            FOR c IN chunks
                FILTER c.node.asset_dir != null
                RETURN c.node.asset_dir
        )

        // Build the final result
        LET result = {{
            text: responseText,
            score: avgScore,
            metadata: {{
                length: LENGTH(responseText),
                source: (
                    FOR c IN chunks
                        COLLECT doc_uuid = c.node.doc_uuid
                        RETURN doc_uuid
                )[0],  // Taking first document UUID as source
                asset_dirs: asset_dirs
            }}
        }}

        // Return the result
        RETURN result
    """
    return aql


def get_gnn_chunk_search_query(collection_name: str):
    """
    Get the AQL query for vector graph search

    Args:
        params (dict, optional): Optional parameters to customize the query

    Returns:
        str: The AQL query string
    """

    aql = f"""
        // Vector search step to find relevant chunks
        LET scoredDocs = (
            FOR doc IN {collection_name}_Chunk
                LET score = COSINE_SIMILARITY(doc.embedding, @query)
                SORT score DESC
                LIMIT @topk_docs
                RETURN {{node: doc, score: score}}
            )

        // Calculate average score directly from scoredDocs
        LET avgScore = AVERAGE(scoredDocs[*].score)

        // Process the chunks
        LET chunks = scoredDocs

        // Fetch entities connected to chunks
        LET chunkEntities = (
            FOR chunkData IN chunks
                LET chunk = chunkData.node

                // For each chunk, find connected entities via HAS_ENTITY edges
                FOR entity IN 1..1 OUTBOUND chunk {collection_name}_HAS_ENTITY
                    FILTER entity.label == "__Entity__"

                    // Group and count entities across chunks
                    COLLECT e = entity WITH COUNT INTO numChunks
                    SORT numChunks DESC
                    LIMIT {VECTOR_GRAPH_SEARCH_ENTITY_LIMIT}

                    LET sim = e.embedding == null ? null  : COSINE_SIMILARITY(e.embedding, @query)

                    LET paths = (
                        RETURN (
                            (sim == null OR ({ARANGO_VECTOR_GRAPH_SEARCH_EMBEDDING_MIN_MATCH} <= sim AND sim <= {ARANGO_VECTOR_GRAPH_SEARCH_EMBEDDING_MAX_MATCH}))
                            ? (
                                // Traverse with depth 1
                                FOR v,edge,p IN 1..1 ANY e {collection_name}_LINKS_TO
                                LIMIT {VECTOR_GRAPH_SEARCH_ENTITY_LIMIT_MINMAX_CASE}
                                RETURN p
                            )
                            : (
                                (sim != null AND sim > {ARANGO_VECTOR_GRAPH_SEARCH_EMBEDDING_MAX_MATCH})
                                ? (
                                    // Traverse with depth 2
                                    FOR v,edge,p IN 1..2 ANY e {collection_name}_LINKS_TO
                                    LIMIT {VECTOR_GRAPH_SEARCH_ENTITY_LIMIT_MAX_CASE}
                                    RETURN p
                                )
                                : (
                                    // Fallback case
                                    [ {{ vertices: [ e ], edges: [] }} ]
                                )
                            )
                        )
                    )
                    RETURN {{ e, paths }}
            )

        // Collect all unique nodes and edges from paths
        LET allPaths = FLATTEN(
            FOR pe IN chunkEntities
                RETURN pe.paths
            )

        // De-duplicate nodes and relationships across chunks
        LET uniqueRelationships = UNIQUE(
            FLATTEN(
                FOR p IN FLATTEN(allPaths)
                    FILTER p.edges != null
                    RETURN p.edges
            )
        )

        // Generate texts from chunks
        LET texts = (
            FOR c IN chunks
                RETURN c.node.text
            )


        // Generate relationship texts - equivalent to the Cypher relTexts logic
        LET relInfo = (
            FOR r IN uniqueRelationships
                LET startNode = DOCUMENT(r._from)
                LET endNode = DOCUMENT(r._to)

                RETURN {{
                    sourceNode : CONCAT(startNode.type, ":", startNode.name),
                    targetNode : CONCAT(endNode.type, ":", endNode.name),
                    relation : r.type
                }}
            )

        // Combine texts into response text - equivalent to the Cypher text formatting
        LET sourceNodes = relInfo[*].sourceNode
        LET targetNodes = relInfo[*].targetNode
        LET relations = relInfo[*].relation

        // Collect asset_dirs from chunks
        LET asset_dirs = (
            FOR c IN chunks
                FILTER c.node.asset_dir != null
                RETURN c.node.asset_dir
        )

        // Build the final result
        LET result = {{
            score: avgScore,
            sourceNodes: sourceNodes,
            targetNodes: targetNodes,
            relations: relations,
            relTexts: texts
        }}

        // Return the result
        RETURN result
    """
    return aql


def get_entity_search_query(collection_name: str):
    """
    Get the AQL query for entity search
    """
    aql = f"""
        // Vector search step to find relevant entities
        LET scoredDocs = (
            FOR doc IN {collection_name}_Entity
                LET score = COSINE_SIMILARITY(doc.embedding, @query)
                SORT score DESC
                LIMIT @topk_docs
                RETURN {{node: doc, score: score}}
        )
        LET avgScore = AVERAGE(scoredDocs[*].score)
        LET entities = scoredDocs

        // Get related chunks
        LET chunks = (
            FOR n IN entities
                LET entity = n.node

                FOR chunk IN 1..1 INBOUND entity {collection_name}_HAS_ENTITY
                    COLLECT c = chunk WITH COUNT INTO freq
                    SORT freq DESC
                    LIMIT {ENTITY_SEARCH_TOP_CHUNKS}
                    RETURN c
        )

        LET entityIds = entities[*].node._id

        LET relationships = (
            FOR n IN entities
                LET entity = n.node
                FOR v, e IN 1..1 OUTBOUND entity {collection_name}_LINKS_TO // TODO: Clarify edge collection
                    FILTER v._id IN entityIds
                    RETURN e
        )

        LET outside = (
            FOR n IN entities
                LET entity = n.node

                FOR v, e IN 1..1 ANY entity {collection_name}_LINKS_TO // TODO: Clarify edge collection
                    FILTER v._id NOT IN entityIds
                    COLLECT m = v INTO rels = e
                    LET freq = LENGTH(rels)
                    SORT freq DESC
                    LIMIT {ENTITY_SEARCH_TOP_OUTSIDE_RELS}

                    RETURN {{rels, m}}
        )

        LET nodeTexts = (
            FOR e IN entities
                LET entity = e.node
                RETURN CONCAT_SEPARATOR(" ", entity.name, entity.description)
        )

        LET responseText = CONCAT(
            "Text Content:\n",
            CONCAT_SEPARATOR("\n----\n", chunks[*].text),
            "\n----\nEntities:\n",
            nodeTexts
        )

        RETURN {{
            text: {{
                chunks: chunks[*].text,
                responseText: responseText,
                entities: nodeTexts,
                relationships: (
                    FOR lt IN relationships // TODO: Determine if just LINKS_TO is needed
                        RETURN CONCAT_SEPARATOR(" ", PARSE_IDENTIFIER(lt._from).key, lt.type, PARSE_IDENTIFIER(lt._to).key)
                ),
                out: {{
                    nodes: (
                        FOR o IN outside
                            LET n = o.m
                            RETURN CONCAT_SEPARATOR(":", n.type, n.name, n.description)
                    ),

                    relationships: (
                        FOR r IN outside
                            LET rel = r.rels
                            FOR r1 in rel
                                LET f = (FOR e IN {collection_name}_Entity FILTER e._id == r1._from RETURN {{name: e.name, type: e.type}})[0]
                                LET t = (FOR e IN {collection_name}_Entity FILTER e._id == r1._to RETURN {{name: e.name, type: e.type}})[0]
                                RETURN CONCAT_SEPARATOR(
                                    ":",
                                    f.type,
                                    f.name,
                                    r1.type,
                                    t.type,
                                    t.name
                                )
                    )
                }}
            }},
            avgScore,
            metadata: {{
                length: LENGTH(responseText),
            }}
        }}
    """
    return aql


def get_retrieval_query(collection_name: str, retriever_type: str) -> str:
    """
    Generate the retrieval query with the user-supplied collection name.

    The query expects:
      - The node collection to be named {collection_name}_node.
      - The graph to be named '{collection_name}' (with quotes as needed).

    Arguments:
      collection_name: The base name provided by the user.

    Returns:
      A string representing the retrieval query.
    TODO: Filter first once ArangoDB supports it.
    """
    if retriever_type == "chunk":
        return get_chunk_search_query(collection_name)
    elif retriever_type == "gnn_chunk":
        return get_gnn_chunk_search_query(collection_name)
    elif retriever_type == "entity":
        return get_entity_search_query(collection_name)
    elif retriever_type == "planner_chunk":
        return get_planner_chunk_search_query(collection_name)
    elif retriever_type == "planner_entity":
        return get_planner_entity_search_query(collection_name)
    elif retriever_type == "subtitle":
        return get_planner_subtitle_search_query(collection_name)
    else:
        logger.warning(
            f"Invalid retriever type: {retriever_type}, Defaulting to chunk search"
        )
        return get_chunk_search_query(collection_name)


def get_adv_chat_template_image(prompt_config: dict, image: bool = False) -> str:
    if image:
        return (
            (
                prompt_config["ADV_CHAT_TEMPLATE_IMAGE"]
                + prompt_config["ADV_CHAT_SUFFIX"]
            )
            if (
                prompt_config
                and prompt_config["ADV_CHAT_TEMPLATE_IMAGE"]
                and prompt_config["ADV_CHAT_SUFFIX"]
            )
            else ADV_CHAT_TEMPLATE_IMAGE + ADV_CHAT_SUFFIX
        )
    else:
        return (
            (prompt_config["ADV_CHAT_TEMPLATE_TEXT"] + prompt_config["ADV_CHAT_SUFFIX"])
            if (
                prompt_config
                and prompt_config["ADV_CHAT_TEMPLATE_TEXT"]
                and prompt_config["ADV_CHAT_SUFFIX"]
            )
            else ADV_CHAT_TEMPLATE_TEXT + ADV_CHAT_SUFFIX
        )


ADV_CHAT_TEMPLATE_IMAGE = """
You are an AI assistant that answers questions based on the provided context.

The context includes retrieved information, relevant chat history, and potentially visual data.
The image context contains images if not empty.
Determine if more visual data (images) would be helpful to answer this question accurately.
For example, if the question is about color of an object, location of an object, or other visual information, visual data is needed.
If image context is not empty, you likely do not need more visual data.
Use all available context to provide accurate and contextual answers.
If the fetched context is insufficient, formulate a better question to
fetch more relevant information. Do not reformulate the question if image data is needed.

You must respond in the following JSON format:
{
    "description": "A description of the answer",\
    "answer": "your answer here or null if more info needed",\
    "updated_question": "reformulated question to get better database results" or null,\
    "confidence": 0.95, // number between 0-1\
    "need_image_data": "true" // string indicating if visual data is needed\
}

Example 1 (when you have enough info from text):
{
    "description": "A description of the answer",\
    "answer": "The worker dropped a box at timestamp 78.0 and it took 39 seconds to remove it",\
    "updated_question": null,\
    "confidence": 0.95,\
    "need_image_data": "false"\
}

Example 2 (when you need visual data):
{
    "description": "A description of the answer",\
    "answer": null,\
    "updated_question": null, //must be null\
    "confidence": 0,\
    "need_image_data": "true"\
}

Example 3 (when you need more context):
{
    "description": "A description of the answer",\
    "answer": null,\
    "updated_question": "What events occurred between timestamp 75 and 80?",\
    "confidence": 0,\
    "need_image_data": "false"\
}

Only respond with valid JSON. Do not include any other text.
"""


ADV_CHAT_TEMPLATE_TEXT = """
You are an AI assistant that answers questions based on the provided context.

The context includes retrieved information and relevant chat history.
Use all available context to provide accurate and contextual answers.
If the fetched context is insufficient, formulate a better question to
fetch more relevant information.

You must respond in the following JSON format:
{
    "description": "A description of the answer",\
    "answer": "your answer here or null if more info needed",\
    "updated_question": "reformulated question to get better database results" or null,\
    "confidence": 0.95 // number between 0-1\
}

Example 1 (when you have enough info from text):
{
    "description": "A description of the answer",\
    "answer": "The worker dropped a box at timestamp 78.0 and it took 39 seconds to remove it",\
    "updated_question": null,\
    "confidence": 0.95\
}

Example 2 (when you need more context):
{
    "description": "A description of the answer",\
    "answer": null,\
    "updated_question": "What events occurred between timestamp 75 and 80?",\
    "confidence": 0\
}

Only respond with valid JSON. Do not include any other text.
"""

ADV_CHAT_SUFFIX = """

When you have enough information, in the "answer" field format your response according to these instructions:

Your task is to provide accurate and comprehensive responses to user queries based on the context, chat history, and available resources.
Answer the questions from the point of view of someone looking at the context.

### Response Guidelines:
1. **Direct Answers**: Provide clear and thorough answers to the user's queries without headers unless requested. Avoid speculative responses.
2. **Utilize History and Context**: Leverage relevant information from previous interactions, the current user input, and the context.
3. **No Greetings in Follow-ups**: Start with a greeting in initial interactions. Avoid greetings in subsequent responses unless there's a significant break or the chat restarts.
4. **Admit Unknowns**: Clearly state if an answer is unknown. Avoid making unsupported statements.
5. **Avoid Hallucination**: Only provide information relevant to the context. Do not invent information.
6. **Response Length**: Keep responses concise and relevant. Aim for clarity and completeness within 4-5 sentences unless more detail is requested.
7. **Tone and Style**: Maintain a professional and informative tone. Be friendly and approachable.
8. **Error Handling**: If a query is ambiguous or unclear, ask for clarification rather than providing a potentially incorrect answer.
9. **Summary Availability**: If the context is empty, do not provide answers based solely on internal knowledge. Instead, respond appropriately by indicating the lack of information.
10. **Absence of Objects**: If a query asks about objects which are not present in the context, provide an answer stating the absence of the objects in the context. Avoid giving any further explanation. Example: "No, there are no mangoes on the tree."
11. **Absence of Events**: If a query asks about an event which did not occur in the context, provide an answer which states that the event did not occur. Avoid giving any further explanation. Example: "No, the pedestrian did not cross the street."
12. **Object counting**: If a query asks the count of objects belonging to a category, only provide the count. Do not enumerate the objects.

### Example Responses:
User: Hi
AI Response: 'Hello there! How can I assist you today?'

User: "What is Langchain?"
AI Response: "Langchain is a framework that enables the development of applications powered by large language models, such as chatbots. It simplifies the integration of language models into various applications by providing useful tools and components."

User: "Can you explain how to use memory management in Langchain?"
AI Response: "Langchain's memory management involves utilizing built-in mechanisms to manage conversational context effectively. It ensures that the conversation remains coherent and relevant by maintaining the history of interactions and using it to inform responses."

User: "I need help with PyCaret's classification model."
AI Response: "PyCaret simplifies the process of building and deploying machine learning models. For classification tasks, you can use PyCaret's setup function to prepare your data. After setup, you can compare multiple models to find the best one, and then fine-tune it for better performance."

User: "What can you tell me about the latest realtime trends in AI?"
AI Response: "I don't have that information right now. Is there something else I can help with?"

**IMPORTANT** : YOUR KNOWLEDGE FOR ANSWERING THE USER'S QUESTIONS IS LIMITED TO THE CONTEXT PROVIDED ABOVE.

Note: This system does not generate answers based solely on internal knowledge. It answers from the information provided in the user's current and previous inputs, and from the context.
"""


QUESTION_ANALYSIS_PROMPT = """Analyze this question and identify key elements for graph database retrieval.
Question: {question}

Identify and return as JSON:
1. Entity types mentioned. Available entity types: {entity_types}
2. Relationships of interest
3. Time references
4. Sort by: "start_time" or "end_time" or "score"
5. Location references
6. Retrieval strategy (similarity, temporal)
    a. similarity: If the question needs to find similar content, return the retrieval strategy as similarity
    b. temporal: If the question is about a specific time range and you can return at least one of the start and end time, then return the strategy as temporal and the start and end time in the time_references field as float or null if not present. Strategy cannot be temporal if both start and end time are not present. The start and end time should be in seconds.

Example response:
{{
    "entity_types": ["Person", "Box"],
    "relationships": ["DROPPED", "PICKED_UP"],
    "time_references": {{
        "start": 60.0,
        "end": 400.0
    }},
    "sort_by": "start_time", // "start_time" or "end_time" or "score"
    "location_references": ["warehouse_zone_A"],
    "retrieval_strategy": "temporal"
}}

Output only valid JSON. Do not include any other text.
"""
