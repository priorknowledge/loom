import loom.compat.format
import loom.runner
import loom.generate
import parsable

commands = [
    loom.compat.format.import_latent,
    loom.compat.format.export_latent,
    loom.compat.format.import_data,
    loom.compat.format.export_log,
    loom.runner.shuffle,
    loom.runner.infer,
    loom.runner.posterior_enum,
    loom.runner.query,
    loom.generate.generate,
]

map(parsable.command, commands)
parsable.dispatch()
