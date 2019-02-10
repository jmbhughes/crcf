import oaforests.ccrf


class TestAxisAlignedRule:
    def test_init(self):
        rule = ccrf.AxisAlignedRule(5, 0)
        assert rule.dimension == 5
        assert rule.value == 0
