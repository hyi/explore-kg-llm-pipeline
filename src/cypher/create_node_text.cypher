MATCH (n)
WHERE n.name IS NOT NULL
SET n.node_text =
  n.name +
  CASE
    WHEN n.description IS NOT NULL AND trim(n.description) <> ""
    THEN " — " + n.description
    ELSE ""
  END;
