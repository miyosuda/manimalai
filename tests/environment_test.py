import unittest
from manimalai.environment import Blackout


class EnvironmentTest(unittest.TestCase):
    def test_blackout(self):
        blackout0 = Blackout([-20])
        for i in range(81):
            ret = blackout0.is_blacked_out(i)
            self.assertEqual(ret, (i % 40) >= 20)

        blackout1 = Blackout([5,10,15,20,25])
        # 5~9:   True
        # 15~19: True
        # 25~:   True

        for i in range(0, 5):
            ret = blackout1.is_blacked_out(i)
            self.assertEqual(ret, False)
        for i in range(5, 10):
            ret = blackout1.is_blacked_out(i)
            self.assertEqual(ret, True)
        for i in range(10, 15):
            ret = blackout1.is_blacked_out(i)
            self.assertEqual(ret, False)
        for i in range(15, 20):
            ret = blackout1.is_blacked_out(i)
            self.assertEqual(ret, True)
        for i in range(20, 25):
            ret = blackout1.is_blacked_out(i)
            self.assertEqual(ret, False)
        for i in range(25, 30):
            ret = blackout1.is_blacked_out(i)
            self.assertEqual(ret, True)
            
        
if __name__ == '__main__':
    unittest.main()
