# Local data and publication

The source release contains application code, placeholder configuration,
synthetic test/replay data and licensed vendor assets. It excludes private robot
keys and serial numbers, account credentials, connection settings, field photos,
point clouds, session logs, diagnostics, handoff documents, archives and local
Python runtimes.

Live operation saves captures and derived maps below `outputs/go2-live-demo/`.
Those files may reveal room geometry, people, camera images, device addresses
and motion history. They are local runtime data and are ignored by Git. The
application has no project analytics or automatic capture-upload feature.
Optional key retrieval explicitly signs in to the selected Unitree cloud.

- Keep `device.json`, `connection.json`, `.env` files and private keys local.
- Use `examples/*.example.json` when documenting configuration.
- Check `git status` and the actual staged diff before sharing changes.
- Run `python work/check_public_source.py` on tracked files.
- Review screenshots and generated reports separately before posting them.

The publication checker catches selected credential patterns and forbidden data
paths; it is not a guarantee that arbitrary future content is anonymized. The
repository's public history starts from the reviewed source snapshot, without
importing deployment directories or private historical archives.
