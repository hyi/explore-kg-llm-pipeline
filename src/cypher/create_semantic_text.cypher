MATCH (s)-[r]->(o)
SET r.semantic_text =
  trim(
    // --- Subject ---
    coalesce(r.llm_subject, s.name, '') +

    CASE
      WHEN r.llm_subject_type IS NOT NULL
        AND trim(toString(r.llm_subject_type)) <> ''
        AND toUpper(trim(toString(r.llm_subject_type))) <> 'NA'
      THEN ' (subject type: ' + trim(toString(r.llm_subject_type)) + ')'
      ELSE ''
    END +

    CASE
      WHEN r.llm_subject_qualifier IS NOT NULL
        AND trim(toString(r.llm_subject_qualifier)) <> ''
        AND toUpper(trim(toString(r.llm_subject_qualifier))) <> 'NA'
        AND trim(toString(r.llm_subject_qualifier)) <> "{'NA': 'NA'}"
        AND trim(toString(r.llm_subject_qualifier)) <> '{}'
      THEN ' (subject qualifier: '
        + trim(toString(r.llm_subject_qualifier))
        + ')'
      ELSE ''
    END +

    // --- Relationship ---
    ' ' + coalesce(r.llm_relationship, type(r), '') + ' ' +

    // --- Object ---
    coalesce(r.llm_object, o.name, '') +

    CASE
      WHEN r.llm_object_type IS NOT NULL
        AND trim(toString(r.llm_object_type)) <> ''
        AND toUpper(trim(toString(r.llm_object_type))) <> 'NA'
      THEN ' (object type: ' + trim(toString(r.llm_object_type)) + ')'
      ELSE ''
    END +

    CASE
      WHEN r.llm_object_qualifier IS NOT NULL
        AND trim(toString(r.llm_object_qualifier)) <> ''
        AND toUpper(trim(toString(r.llm_object_qualifier))) <> 'NA'
        AND trim(toString(r.llm_object_qualifier)) <> "{'NA': 'NA'}"
        AND trim(toString(r.llm_object_qualifier)) <> '{}'
      THEN ' (object qualifier: '
        + trim(toString(r.llm_object_qualifier))
        + ')'
      ELSE ''
    END +

    // --- Claim-level qualification ---
    CASE
      WHEN r.llm_statement_qualifier IS NOT NULL
        AND trim(toString(r.llm_statement_qualifier)) <> ''
        AND toUpper(trim(toString(r.llm_statement_qualifier))) <> 'NA'
        AND trim(toString(r.llm_statement_qualifier)) <> "{'NA': 'NA'}"
        AND trim(toString(r.llm_statement_qualifier)) <> '{}'
      THEN ' (statement qualifier: '
        + trim(toString(r.llm_statement_qualifier))
        + ')'
      ELSE ''
    END
  );
