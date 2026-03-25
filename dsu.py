class DSU:
    """
    Disjoint Set Union / Union-Find.

    Нужен для объединения совпавших изображений в группы.
    Если A совпал с B, а B совпал с C, то все они окажутся в одной компоненте.
    """

    def __init__(self, n: int):
        if n < 0:
            raise ValueError("n must be >= 0")
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        """
        Находит корень множества с path compression.
        """
        if x < 0 or x >= len(self.parent):
            raise IndexError(f"Index out of range: {x}")

        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: int, b: int) -> bool:
        """
        Объединяет множества a и b.
        Возвращает True, если реально объединили, иначе False.
        """
        ra = self.find(a)
        rb = self.find(b)

        if ra == rb:
            return False

        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1

        return True

    def groups(self) -> dict[int, list[int]]:
        """
        Возвращает группы в виде:
        {root: [indices]}
        """
        result: dict[int, list[int]] = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            result.setdefault(root, []).append(i)
        return result