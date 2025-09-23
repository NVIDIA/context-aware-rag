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


"""planner_constants.py: File contains constants for graph rag"""

SUBTITLE_SEARCH_QUERY_FORMATTED = """
    WITH node as subtitle, score
    WITH collect(DISTINCT {subtitle: subtitle, score: score}) AS subtitles, avg(score) as avg_score
    RETURN { subtitles: [s IN subtitles | {text: s.subtitle.text + " ( start_time: " + s.subtitle.start_time + " end_time: " + s.subtitle.end_time + " )"}]} AS text,
    avg_score AS score,
    { length: 2 } AS metadata
"""


CHUNK_SEARCH_QUERY = """
    WITH node as chunk, score
    // find the document of the chunk
    MATCH (chunk)-[:PART_OF]->(d:Document)
    WHERE CASE
        WHEN $uuid IS NOT NULL THEN d.uuid = $uuid
        ELSE true
    END

    // aggregate chunk-details
    WITH d, collect(DISTINCT {chunk: chunk, score: score}) AS chunks, avg(score) as avg_score
    WITH d, chunks, avg_score,
     [c IN chunks | "Text: " + c.chunk.text + " ( start_time: " + c.chunk.start_time + " end_time: " + c.chunk.end_time + " chunk_id: " + id(c.chunk) + " camera_id: " + coalesce(c.chunk.camera_id, "N/A") + " )"] AS textsFormatted,
     [c IN chunks | {id: c.chunk.id, camera_id: coalesce(c.chunk.camera_id, "N/A"), score: c.score, chunkIdx: c.chunk.chunkIdx, content: c.chunk.text}] AS chunkdetails

    // Combine texts into response text
    WITH d, avg_score, chunkdetails,
        "\n" + apoc.text.join(textsFormatted, "\n") + "\n" AS text
    RETURN
        text,
        avg_score AS score,
        {
            length: size(text),
            source: d.uuid,
            chunkdetails: chunkdetails
        } AS metadata
    """


###Entity search
ENTITY_SEARCH_TOP_K = 10
ENTITY_SEARCH_TOP_CHUNKS = 5

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
     }} AS chunks

"""

ENTITY_SEARCH_QUERY_SUFFIX = """
RETURN {
  chunks: [c IN chunks | {text: c.text + " ( start_time: " + c.start_time + " end_time: " + c.end_time + " chunk_id: " + id(c) + " camera_id: " + coalesce(c.camera_id, "N/A") + " )"}],
  entities: [
    n IN nodes |
    CASE
      WHEN size(labels(n)) > 1 THEN
        apoc.coll.removeAll(labels(n), ["__Entity__"])[0] + ":" + coalesce(n.name, n.id) + " " + coalesce(n.description, "") + " ( id: " + id(n) + " )" + " ( camera_id: [" + apoc.text.join([x IN apoc.coll.flatten([coalesce(n.camera_id, [])]) | toString(x)], ", ") + "] )"
      ELSE
        coalesce(n.name, n.id) + " " + coalesce(n.description, "") + " ( entity_id: " + id(n) + " )" + " ( camera_id: [" + apoc.text.join([x IN apoc.coll.flatten([coalesce(n.camera_id, [])]) | toString(x)], ", ") + "] )"
    END
  ]
} AS text,
  score,
  {entities: metadata} AS metadata
"""


PLANNER_ENTITY_SEARCH_QUERY = (
    ENTITY_SEARCH_QUERY.format(
        topChunks=ENTITY_SEARCH_TOP_CHUNKS,
    )
    + ENTITY_SEARCH_QUERY_SUFFIX
)


def get_planner_chunk_search_query(collection_name: str):
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

    // Format texts for each chunk
    LET textsFormatted = (
        FOR chunk IN chunks
            LET c = chunk.node
            RETURN CONCAT(
                "Text: ", c.text,
                " ( start_time: ", TO_STRING(c.start_time),
                " end_time: ", TO_STRING(c.end_time),
                " chunk_id: ", c._key,
                " camera_id: ", (c.camera_id || "N/A"), " )"
            )
    )

    LET chunkdetails = (
        FOR chunk IN chunks
            LET c = chunk.node
            RETURN {{
                id: c._key,
                camera_id: (c.camera_id || "N/A"),
                score: c.score,
                chunkIdx: c.chunkIdx,
                content: c.text
            }}
    )

    // Combine texts into response text
    LET text = CONCAT("\\n", CONCAT_SEPARATOR("\\n", textsFormatted), "\\n")

    RETURN {{
        text: text,
        score: avgScore,
        metadata: {{
            length: LENGTH(text),
            source: chunks[0].node.uuid,
            chunkdetails: chunkdetails
        }}
    }}
    """
    return aql


def get_planner_entity_search_query(collection_name: str):
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

        LET chunksTexts = (
            FOR c IN chunks
                RETURN (
                    CONCAT(
                        c.text,
                        " ( start_time: ", TO_STRING(c.start_time),
                        " end_time: ", TO_STRING(c.end_time),
                        " chunk_id: ", c._key,
                        " camera_id: ", (c.camera_id || "N/A"), " )"
                    )
                )
            )

        LET entitiesTexts =  (
            FOR entity IN entities
                RETURN (
                    LET n = entity.node
                    LET camera_ids_flattened = (
                        IS_ARRAY(n.camera_id) ?
                        FLATTEN([n.camera_id]) :
                        (n.camera_id ? [n.camera_id] : [])
                    )
                    LET camera_ids_string = CONCAT_SEPARATOR(
                        ", ", (FOR x IN camera_ids_flattened RETURN TO_STRING(x))
                    )

                    RETURN (
                        CONCAT(
                            n.type, ":",
                            (n.name || n.id || ""), " ",
                            (n.description || ""),
                            " ( id: ", n._key, " )",
                            " ( camera_id: [", camera_ids_string, "] )"
                        )
                    )
                )
            )
        LET textformatted = CONCAT("chunks: ", chunksTexts, "\n\n", "entities: ", entitiesTexts)
        RETURN {{
            text: textformatted,
            score: avgScore,
            metadata: {{
                entities: entitiesTexts
            }}
        }}
    """
    return aql


def get_planner_subtitle_search_query(collection_name: str):
    return ""
