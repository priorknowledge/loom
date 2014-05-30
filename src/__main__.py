import loom.format
import loom.runner
import loom.generate
import parsable

commands = [
    loom.format.import_latent,
    loom.format.export_latent,
    loom.format.import_data,
    loom.format.export_log,
    loom.runner.shuffle,
    loom.runner.infer,
    loom.runner.posterior_enum,
    loom.runner.predict,
    loom.generate.generate,
]

map(parsable.command, commands)
parsable.dispatch()
