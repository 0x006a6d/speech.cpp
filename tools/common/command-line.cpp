#include "command-line.h"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <stdexcept>

#include "spelling.h"
#include "terminal.h"

namespace {

const Flag * find_flag(const Command & command, const std::string & name) {
    for (const Flag & f : command.flags) {
        if (f.name == name) return &f;
    }
    return nullptr;
}

/** The request option whose flag is `name`, or false. */
bool option_of_flag(const std::string & name, speech_option & option) {
    for (speech_option o : vocabulary()) {
        if (option_flag(o) == name) {
            option = o;
            return true;
        }
    }
    return false;
}

/** "; did you mean --voice?" where `name` is a slip of one of the subcommand's flags, or "". */
std::string meant(const Command & command, const std::string & name) {
    std::vector<std::string> flags = {"--help"};
    for (const Flag & f : command.flags) flags.push_back(f.name);
    if (command.request_options) {
        for (speech_option o : vocabulary()) flags.push_back(option_flag(o));
    }
    // A flag with a task's prefix is compared with the flags of that prefix.
    const size_t dot = name.find('.');
    if (command.task_options && dot != std::string::npos) {
        flags.clear();
        for (speech_task t : {SPEECH_TASK_SYNTHESIS, SPEECH_TASK_RECOGNITION, SPEECH_TASK_DETECTION}) {
            for (speech_option o : vocabulary()) flags.push_back(std::string("--") + task_prefix(t) + "." + option_flag(o).substr(2));
        }
    }
    const std::string near = nearest(name, flags);
    return near.empty() ? "" : "; did you mean " + near + "?";
}

std::string padded(const std::string & s, size_t width) {
    return s.size() >= width ? s + "  " : s + std::string(width - s.size(), ' ');
}

}  // namespace

bool CommandLine::has(const std::string & flag) const {
    return std::any_of(given_.begin(), given_.end(), [&](const Given & g) { return g.flag == flag; });
}

std::optional<std::string> CommandLine::value(const std::string & flag) const {
    for (const Given & g : given_) {
        if (g.flag == flag) return g.value;
    }
    return std::nullopt;
}

std::vector<std::string> CommandLine::values(const std::string & flag) const {
    std::vector<std::string> out;
    for (const Given & g : given_) {
        if (g.flag == flag) out.push_back(g.value);
    }
    return out;
}

std::optional<int> CommandLine::integer(const std::string & flag) const {
    const std::optional<std::string> text = value(flag);
    if (!text) return std::nullopt;
    const size_t first = !text->empty() && (*text)[0] == '-' ? 1 : 0;
    const bool digits = text->size() > first && text->find_first_not_of("0123456789", first) == std::string::npos;
    errno = 0;
    const long long v = digits ? std::strtoll(text->c_str(), nullptr, 10) : 0;
    if (!digits || errno == ERANGE || v < INT_MIN || v > INT_MAX) {
        throw UsageError(flag + " takes a whole number, not \"" + *text + "\"");
    }
    return (int) v;
}

Loading CommandLine::loading(bool warmup, const std::optional<std::string> & device) const {
    Loading out;
    out.device = value("--device");
    if (!out.device) out.device = device;
    out.threads = integer("--threads");
    out.warmup = warmup && !has("--no-warmup");
    for (const std::string & v : values("--add-voice")) {
        const size_t eq = v.find('=');
        if (eq == std::string::npos || eq == 0 || eq + 1 == v.size()) {
            throw UsageError("--add-voice takes NAME=FILE, a voice's name and its voice file or WAVE file, not \"" + v + "\"");
        }
        out.voices.push_back({v.substr(0, eq), v.substr(eq + 1)});
    }
    return out;
}

CommandLine parse_command_line(const Command & command, const std::vector<std::string> & args) {
    CommandLine line(command);
    bool flags_end = false;
    for (size_t i = 0; i < args.size(); i++) {
        const std::string & arg = args[i];
        if (flags_end || arg.size() < 2 || arg[0] != '-') {
            line.args.push_back(arg);
            continue;
        }
        if (arg == "--") {
            flags_end = true;
            continue;
        }
        const size_t eq = arg.compare(0, 2, "--") == 0 ? arg.find('=') : std::string::npos;
        const std::string name = arg.substr(0, eq);
        const std::optional<std::string> attached = eq == std::string::npos ? std::nullopt : std::optional<std::string>(arg.substr(eq + 1));
        speech_option option;
        const Flag * flag = find_flag(command, name);
        std::optional<speech_task> task;
        bool request_option = !flag && command.request_options && option_of_flag(name, option);
        if (!flag && !request_option && command.task_options) {
            const size_t dot = name.find('.');
            for (speech_task t : {SPEECH_TASK_SYNTHESIS, SPEECH_TASK_RECOGNITION, SPEECH_TASK_DETECTION}) {
                if (dot != std::string::npos && name.compare(0, dot, std::string("--") + task_prefix(t)) == 0 &&
                    option_of_flag("--" + name.substr(dot + 1), option)) {
                    task = t;
                    request_option = true;
                }
            }
        }
        if (!flag && !request_option) throw UsageError("speech " + command.name + " has no flag " + name + meant(command, name));
        // A boolean option is true alone and takes true or false only after "=", so that the argument after it stays
        // an argument.
        const bool boolean = request_option && speech_option_type(option) == SPEECH_TYPE_BOOL;
        const bool takes_value = flag ? !flag->value.empty() : !boolean;
        std::string value = boolean ? attached.value_or("true") : "";
        if (takes_value) {
            if (attached) {
                value = *attached;
            } else if (i + 1 < args.size()) {
                value = args[++i];
            } else {
                throw UsageError(name + " needs a value");
            }
        } else if (attached && !boolean) {
            throw UsageError(name + " takes no value; give it alone");
        }
        if (request_option) {
            for (const RequestOption & o : line.options) {
                if (!task && o.option == option) throw UsageError(name + " is given twice; give it once");
            }
            for (const TaskOption & o : line.task_options) {
                if (task && o.task == *task && o.option.option == option) throw UsageError(name + " is given twice; give it once");
            }
            try {
                const RequestOption read{option, option_from_text(option, value)};
                if (task) line.task_options.push_back({*task, read});
                else line.options.push_back(read);
            } catch (const std::invalid_argument & e) {
                throw UsageError(e.what());
            }
            continue;
        }
        if (!flag->repeatable && line.has(name)) throw UsageError(name + " is given twice; give it once");
        line.given_.push_back({name, value});
    }
    if (line.args.size() < command.min_args) {
        if (command.model && (line.args.empty() || !names_a_model(line.args[0]))) throw UsageError(no_model_message(command.usage, *command.model));
        throw UsageError("an argument is missing: speech " + command.usage);
    }
    if (line.args.size() > command.max_args) {
        throw UsageError("there are more arguments than speech " + command.usage + " takes; quote an argument that holds spaces");
    }
    return line;
}

bool asks_for_help(const std::vector<std::string> & args) {
    for (const std::string & a : args) {
        if (a == "--") return false;
        if (a == "--help" || a == "-h") return true;
    }
    return false;
}

std::string command_help(const Command & command, FILE * stream) {
    using terminal::Color;
    using terminal::paint;
    std::string out = paint(Color::Bold, "usage:", stream) + " speech " + command.usage + "\n\n" + command.description + "\n";
    if (command.model) out += "MODEL is a model file (.gguf), or the NAME[:TYPE] of one in `speech models`, fetched the first time it is used.\n";
    constexpr size_t kWidth = 28;
    if (!command.flags.empty()) out += "\n";
    for (const Flag & f : command.flags) {
        out += "  " + paint(Color::Lavender, padded(f.name + (f.value.empty() ? "" : " " + f.value), kWidth), stream) + f.help +
               (f.repeatable ? "; repeatable" : "") + "\n";
    }
    if (command.request_options) {
        out += "\n" + paint(Color::Bold, "Request options", stream) + ", each the C API's option of the same name; the model refuses one it does not take,\n"
               "and `speech info MODEL` lists those it takes with their defaults and ranges:\n";
        for (speech_option o : vocabulary()) {
            out += "  " + paint(Color::Lavender, option_flag(o) + (speech_option_type(o) == SPEECH_TYPE_BOOL ? "[=false]" : " " + std::string(option_metavar(o))),
                                stream) + "\n";
        }
        if (command.task_options) {
            out += "Each also after tts., asr. or vad. (--asr.language), for the synthesis, recognition or detection model alone.\n";
        }
    }
    return out;
}

const char * task_prefix(speech_task task) {
    switch (task) {
        case SPEECH_TASK_SYNTHESIS: return "tts";
        case SPEECH_TASK_RECOGNITION: return "asr";
        case SPEECH_TASK_DETECTION: return "vad";
    }
    return "";
}

Flag device_flag(const std::string & help) {
    return {"--device", "NAME", false, help};
}

Flag threads_flag() {
    return {"--threads", "N", false, "the CPU's threads, 1 or more; the machine's performance cores unless given, or 4 or 8 for the decoding on the CPU of a FastConformer model on a GPU"};
}

Flag add_voice_flag() {
    return {"--add-voice", "NAME=FILE", true, "add a voice from a voice file or a WAVE file before the first request"};
}

Flag no_warmup_flag() {
    return {"--no-warmup", "", false, "skip the short request that compiles a GPU's kernels before the first one"};
}

Flag verbose_flag(const std::string & help) {
    return {"-v", "", false, help};
}
