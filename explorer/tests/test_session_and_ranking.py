import unittest

from explorer.backend.exploration_session import ExplorationSession
from explorer.backend.models import Edge, Node, Path
from explorer.backend.path_ranking import PathRankingService


def make_path(path_id: str, score: float) -> Path:
    node_a = Node(element_id=f"{path_id}-a", id="a", name="A")
    node_b = Node(element_id=f"{path_id}-b", id="b", name="B")
    edge = Edge(
        element_id=f"{path_id}-e",
        type="biolink:related_to",
        start_element_id=node_a.element_id,
        end_element_id=node_b.element_id,
    )
    return Path(id=path_id, nodes=[node_a, node_b], edges=[edge], score=score)


class SessionAndRankingTest(unittest.TestCase):
    def test_ranking_uses_similarity_score_descending(self) -> None:
        low = make_path("low", 0.2)
        high = make_path("high", 0.9)

        ranked = PathRankingService().rank([low, high])

        self.assertEqual(["high", "low"], [path.id for path in ranked])

    def test_session_accept_reject_bookmark_and_notes(self) -> None:
        path = make_path("path-1", 0.8)
        session = ExplorationSession()

        session.bookmark(path)
        session.accept(path)
        session.save_note(path.id, "important")
        session.reject(path)

        self.assertNotIn(path.id, session.accepted_paths)
        self.assertNotIn(path.id, session.bookmarked_paths)
        self.assertIn(path.id, session.rejected_paths)
        self.assertEqual("important", session.user_notes[path.id])


if __name__ == "__main__":
    unittest.main()
