"""Forward virtual-stick axes and neutralization; no robot transport."""
import types
import unittest

import go2_live_demo as app
from test_go2_navigation import NavigationDemo


class NavigationChannelTests(unittest.IsolatedAsyncioTestCase):
    def demo(self):
        demo, sent = NavigationDemo(), []
        demo.conn = types.SimpleNamespace(datachannel=types.SimpleNamespace(
            channel=types.SimpleNamespace(readyState='open'),
            pub_sub=types.SimpleNamespace(publish_without_callback=lambda *args: sent.append(args))))
        demo.move = types.MethodType(app.Demo.move, demo)
        demo.move_joystick = types.MethodType(app.Demo.move_joystick, demo)
        return demo, sent

    async def test_forward_stick_and_stop_neutralize_both_axes(self):
        demo, sent = self.demo()
        app.Demo.move_navigation(demo, .1, .2)
        self.assertEqual(sent[-1][0], app.RTC_TOPIC['WIRELESS_CONTROLLER'])
        axes = sent[-1][1]
        self.assertAlmostEqual(axes['ly'], .30)
        self.assertAlmostEqual(axes['rx'], -.1*app.RIGHT_STICK_PER_YAW)
        self.assertEqual((axes['lx'], axes['ry'], axes['keys']), (0., 0., 0))
        demo.move(0.)
        self.assertEqual(sent[-2][1], {'lx': 0., 'ly': 0., 'rx': 0., 'ry': 0., 'keys': 0})
        self.assertEqual(demo.joystick_ly, 0.)



if __name__ == '__main__':
    unittest.main()
