import unittest

from manimalai.environment import Blackout, AAIEnvironment
from manimalai.arena_config import Vector3


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
            
            
    def test_convert_pos(self):
        env = AAIEnvironment()
        
        pos = Vector3(1, 2, 3)
        p0 = env._convert_pos(pos, offset_y=0.0)
        p1 = env._convert_pos_inv(p0)

        self.assertAlmostEqual(pos.x, p1[0])
        self.assertAlmostEqual(pos.y, p1[1])
        self.assertAlmostEqual(pos.z, p1[2])
                
        env.close()
        
        
    def test_convert_rot(self):
        env = AAIEnvironment()

        r0 = env._convert_rot(358)
        r1 = env._convert_rot_inv(r0)
        
        self.assertAlmostEqual(r1, 358)
        
        env.close()
        
if __name__ == '__main__':
    unittest.main()
