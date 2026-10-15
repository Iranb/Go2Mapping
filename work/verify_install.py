"""Check installed modules and web assets without connecting to a robot."""
import asyncio
from pathlib import Path

from aiohttp.test_utils import TestClient, TestServer
import go2_live_demo as app


async def main():
    root = Path(__file__).resolve().parent.parent
    assert (root/"outputs/go2-live-demo/vendor/three.module.js").is_file()
    demo = app.Demo()
    site = app.create_app(demo)
    site.on_startup.clear()
    site.on_shutdown.clear()
    async with TestClient(TestServer(site)) as client:
        for route in ("/", "/api/status", "/vendor/three.module.js", "/assets/photo-depth.js", "/assets/panorama-view.js", "/assets/spatial-photos.js"):
            response = await client.get(route)
            assert response.status == 200, route
            await response.read()
        response = await client.get("/api/status")
        status = await response.json()
        assert not status["connected"] and not status["active"]
    print("Dependency imports, web assets and status API passed. No robot connection created.")


if __name__ == "__main__":
    asyncio.run(main())
