MATCH (s)-[r]->(o)
SET r.semantic_text =
  trim(
    // --- Core claim ---
    coalesce(r.llm_subject, s.name, '') +

    CASE
      WHEN r.llm_subject_qualifier IS NOT NULL
        AND r.llm_subject_qualifier <> 'NA'
        AND r.llm_subject_qualifier <> "{'NA': 'NA'}"
      THEN ' (subject qualifier: ' + toString(r.llm_subject_qualifier) + ')'
      ELSE ''
    END +

    ' ' + coalesce(r.llm_relationship, type(r), '') + ' ' +

    coalesce(r.llm_object, o.name, '') +

    CASE
      WHEN r.llm_object_qualifier IS NOT NULL
        AND r.llm_object_qualifier <> 'NA'
        AND r.llm_object_qualifier <> "{'NA': 'NA'}"
      THEN ' (object qualifier: ' + toString(r.llm_object_qualifier) + ')'
      ELSE ''
    END +

    // --- Literature context ---
    CASE
      WHEN r.abstract_title IS NOT NULL OR r.abstract_text IS NOT NULL
      THEN
        '\n\nEvidence:' +

        CASE
          WHEN r.abstract_title IS NOT NULL
          THEN '\nTitle: ' + r.abstract_title
          ELSE ''
        END +

        CASE
          WHEN r.abstract_text IS NOT NULL
          THEN '\nAbstract: ' + substring(r.abstract_text, 0, 800)
          ELSE ''
        END
      ELSE ''
    END
  );

