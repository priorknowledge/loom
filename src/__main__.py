import loom.runner
import loom.format
import parsable

commands = [
    loom.format.import_latent,
    loom.format.export_latent,
    loom.format.import_data,
    loom.format.export_data,
    loom.runner.infer,
]

map(parsable.command, commands)
parsable.dispatch()
