"""Checked partial routes, renewed observations, and final-goal arrival."""
import asyncio
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import go2_live_demo as app
from go2_navigation import ARRIVAL, MIN_APPROACH_PROGRESS, GroundGrid
from test_go2_navigation import NavigationDemo, scene
from test_go2_motion_recovery import Clock

class ApproachPlannerTests(unittest.TestCase):

    def test_no_partial_route_without_meaningful_progress(self):
        grid=GroundGrid(scene(wall=[(.5,y*.05) for y in range(-60,61)]),[0,0,.3],0)
        with self.assertRaises(ValueError):
            grid.plan([1.5,0],True)

    def test_complete_route_is_preferred_when_detour_exists(self):
        grid=GroundGrid(scene(wall=[(.8,y*.05) for y in range(-7,8)]),[0,0,.3],0)
        plan=grid.plan([1.8,0],True)
        self.assertFalse(plan['partial'])
        self.assertLessEqual(plan['goal_offset_m'],ARRIVAL)
        self.assertTrue(grid.route_safe(plan['path'],0))


class ApproachControllerTests(unittest.IsolatedAsyncioTestCase):
    async def run_case(self, opens):
        demo,clock,real_sleep=NavigationDemo(),Clock(),asyncio.sleep
        wall=scene(wall=[(1.6,y*.05) for y in range(-60,61)])
        demo.cloud=wall
        original_grid=demo.navigation_grid
        async def update_grid(yaw):
            if opens and demo.sport['position'][0]>.8:
                demo.cloud=scene()
            return await original_grid(yaw)
        demo.navigation_grid=update_grid
        with patch.object(app,'time',clock.namespace()):
            demo.refresh()
            route=GroundGrid(wall,demo.sport['position'],0).plan([2.5,0],True)
            self.assertTrue(route['partial'])
            plan={**route,'target':[2.5,0],'yaw':0,'epoch':0,'start':[0,0], 'used':True,'state':'checking'}
            async def tick(seconds):
                clock.value+=seconds
                demo.refresh()
                await real_sleep(0)
            with patch.object(app.asyncio,'sleep',tick):
                await demo.run_navigation('operator',plan)
        return demo

    async def test_rescans_at_partial_endpoint_then_completes_original_goal(self):
        demo=await self.run_case(True)
        self.assertEqual(demo.phase,'complete',demo.message)
        self.assertLessEqual(demo.report['distance_m'],ARRIVAL)
        self.assertTrue(any(r['trigger']['kind']=='approach' for r in demo.report['replans']))
        self.assertFalse(demo.navigation['partial'])
        self.assertTrue(demo.report['corners'])

    async def test_persistent_blockage_stops_without_false_arrival(self):
        demo=await self.run_case(False)
        self.assertEqual(demo.phase,'error',demo.message)
        self.assertIn('verified endpoint',demo.message)
        self.assertGreater(demo.report['distance_m'],ARRIVAL)
        self.assertLess(demo.sport['position'][0],1.6-.35)
        self.assertIn('stop_response',demo.report)
        self.assertEqual(demo.report['replan_failure']['trigger']['kind'],'approach')

if __name__=='__main__':
    unittest.main()
