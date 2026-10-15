"""Arrival regions and vertical clearance, including a recorded stationary scan."""
import asyncio
import pickle
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import go2_live_demo as app
from go2_demo_core import collision_ceiling, rotation_obstacles
from go2_navigation import GroundGrid, PlanningError, HEADINGS, ARRIVAL, GOAL_REGION
from test_go2_navigation import NavigationDemo, scene
import test_go2_navigation as navigation_tests
from test_go2_motion_recovery import Clock


def niche():
    wall = [(1.5, y*.05) for y in range(-60, 61)]
    wall += [(x*.05, y) for x in range(-30, 31) for y in (-.35, .35)]
    return scene(wall=wall)


def hanging_wall(height):
    return np.vstack((scene(), [[.975, y*.05-.025, height-.025] for y in range(-60, 61)]))


class ReachabilityPlannerTests(unittest.TestCase):
    def test_planning_error_survives_worker_serialization(self):
        error = PlanningError('Target ground coverage incomplete', {'kind': 'unknown', 'point': [1., 0.]})
        restored = pickle.loads(pickle.dumps(error))
        self.assertEqual(str(restored), str(error))
        self.assertEqual(restored.issue, error.issue)

    def test_overhead_surface_is_not_projected_to_the_floor(self):
        cloud = hanging_wall(.70)
        grid = GroundGrid(cloud, [0, 0, .3], 0)
        self.assertLess(grid.ceiling, .60)
        plan = grid.plan([1., 0])
        self.assertEqual(plan['walk_count'], 1)
        self.assertLessEqual(plan['goal_offset_m'], GOAL_REGION)
        occupied, _ = rotation_obstacles(cloud, [0, 0, .3], .05)
        self.assertEqual(len(occupied), 0)

    def test_low_overhang_and_table_leg_still_block(self):
        for cloud in (hanging_wall(.55), np.vstack((hanging_wall(.70), [[.975, -.025, .225]]))):
            grid = GroundGrid(cloud, [0, 0, .3], 0)
            with self.assertRaises(PlanningError) as caught:
                grid.plan([1., 0])
            self.assertEqual(caught.exception.issue['kind'], 'occupied')
            occupied, _ = rotation_obstacles(cloud, [0, 0, .3], .05)
            self.assertGreater(len(occupied), 0)

    def test_ceiling_follows_body_height_and_floor(self):
        low = GroundGrid(hanging_wall(.70), [0, 0, .3], 0)
        high = GroundGrid(hanging_wall(.70), [0, 0, .48], 0)
        self.assertFalse(low.occupied[low.cell([1., 0])[::-1]])
        self.assertTrue(high.occupied[high.cell([1., 0])[::-1]])
        self.assertGreaterEqual(collision_ceiling(.20, .10, .05), .50+.02+.025)

    def test_overhead_does_not_supply_missing_ground(self):
        cloud = hanging_wall(.70)
        centers = cloud[:, :2]+.025
        cloud = cloud[~((np.linalg.norm(centers-[1., 0], axis=1) < .6) & (cloud[:, 2] < .1))]
        grid = GroundGrid(cloud, [0, 0, .3], 0)
        with self.assertRaises(PlanningError) as caught:
            grid.plan([1., 0])
        self.assertEqual(caught.exception.issue['kind'], 'unknown')

    def test_reachable_arrival_region_can_replace_blocked_exact_click(self):
        grid = GroundGrid(niche(), [0, 0, .3], 0)
        goal = [1.05, 0]
        self.assertFalse(any(grid.pose_safe(goal, h) for h in HEADINGS))
        plan = grid.plan(goal)
        self.assertGreater(plan['goal_offset_m'], .04)
        self.assertLessEqual(plan['goal_offset_m'], GOAL_REGION)
        self.assertEqual(plan['arrival_target'], plan['path'][-1])
        self.assertTrue(grid.route_safe(plan['path'], 0))

    def test_goal_adjustment_cannot_cross_the_allowed_region(self):
        grid = GroundGrid(niche(), [0, 0, .3], 0)
        with self.assertRaises(PlanningError):
            grid.plan([1.25, 0])



    def test_already_inside_region_has_no_motion_actions(self):
        grid = GroundGrid(scene(), [0, 0, .3], 0)
        for goal in ([0., 0.], [.05, 0.], [.15, 0.], [.19, 0.]):
            plan = grid.plan(goal)
            self.assertEqual(plan['path'], [[0., 0.]])
            self.assertEqual(plan['actions'], [])
            self.assertEqual(plan['length_m'], 0.)


class ArrivalRegionControllerTests(unittest.IsolatedAsyncioTestCase):
    async def test_adjusted_endpoint_stops_within_arrival_region(self):
        demo, clock, real_sleep = NavigationDemo(), Clock(), asyncio.sleep
        demo.cloud = niche()
        with patch.object(app, 'time', clock.namespace()):
            demo.refresh()
            route = GroundGrid(demo.cloud, demo.sport['position'], 0).plan([1.05, 0])
            plan = {**route, 'target': [1.05, 0], 'yaw': 0, 'epoch': 0,
                    'start': [0, 0], 'used': True, 'state': 'checking'}
            async def tick(seconds):
                clock.value += seconds
                demo.refresh()
                await real_sleep(0)
            with patch.object(app.asyncio, 'sleep', tick):
                await demo.run_navigation('operator', plan)
        self.assertEqual(demo.phase, 'complete', demo.message)
        self.assertLessEqual(demo.report['distance_m'], ARRIVAL)
        self.assertGreater(demo.report['goal_offset_m'], .04)
        self.assertLessEqual(demo.report['goal_offset_m'], GOAL_REGION)
        self.assertIn('stop_response', demo.report)

    async def test_already_nearby_completes_without_motion_commands(self):
        from go2_navigation_sim import simulate
        result, _, commands = await simulate(scene(), [0, 0, .3], 0, [.15, 0])
        self.assertEqual(result['outcome'], 'complete', result['message'])
        self.assertLessEqual(result['final_distance_m'], ARRIVAL)
        self.assertFalse(any(c['vx'] or c['vy'] or c['wz'] for c in commands))



class ReachabilityEndpointTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = navigation_tests.EndpointTests.asyncSetUp
    asyncTearDown = navigation_tests.EndpointTests.asyncTearDown

    async def test_unknown_goal_exposes_missing_ground_beyond_checked_approach(self):
        self.demo.cloud = scene(extent=.7)
        response = await self.client.post('/api/plan', json={'client_id': 'operator', 'target': [1.5, 0]})
        self.assertEqual(response.status, 200, await response.text())
        plan = await response.json()
        self.assertTrue(plan['reachable'])
        self.assertTrue(plan['partial'])
        self.assertGreater(plan['goal_offset_m'], ARRIVAL)
        self.assertEqual(plan['remaining_obstruction']['kind'], 'unknown')
        self.assertIn('full route unverified', plan['reason'])
        grid = GroundGrid(self.demo.cloud, self.demo.sport['position'], 0)
        self.assertTrue(grid.route_safe(plan['path'], 0))
        self.assertEqual(self.demo.translations, [])
        self.assertIn('collision_ceiling_m', plan)


if __name__ == '__main__':
    unittest.main()
