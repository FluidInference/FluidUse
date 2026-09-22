import Foundation

/// `FluidUseLaya answer|tetris|2048|benchmark …` — on-device decision models and demos.
let arguments = Array(CommandLine.arguments.dropFirst())
switch arguments.first {
case "answer":
    await LayaCommand.run(arguments: Array(arguments.dropFirst()))
case "tetris":
    await LayaTetrisCommand.run(arguments: Array(arguments.dropFirst()))
case "2048":
    await Game2048Command.run(arguments: Array(arguments.dropFirst()))
case "benchmark":
    await LayaBenchmarkCommand.run(arguments: Array(arguments.dropFirst()))
default:
    print(
        """
        usage: swift run -c release FluidUseLaya answer --state TEXT --type noul|choice|score --instructions TEXT [--options a|b]
               swift run -c release FluidUseLaya tetris [--policy laya|gliclass|heuristic|random] [--pieces N] [--seed N]
               swift run -c release FluidUseLaya 2048 [--policy laya|laya-choice|gliclass|heuristic|random] [--games N]
               swift run -c release FluidUseLaya benchmark --suites suites.jsonl [--reference reference-rows.jsonl]
        Add --help to any subcommand for its options.
        """)
}
