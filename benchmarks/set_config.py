"""Sets CONFIG_ options in kws_s3/sdkconfig in place (the file also holds the Wi-Fi credentials, which are never
read or printed here). Then run `idf.py reconfigure` and `ninja -C build -j 3`.

    python set_config.py KWS_PROFILE_OPS=y KWS_TELEMETRY_MS=0 ESP_WIFI_IRAM_OPT=n
    python set_config.py --show KWS_TELEMETRY_MS ESP_WIFI_IRAM_OPT
"""
import os
import re
import sys

SDK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kws_s3", "sdkconfig")
SECRET = re.compile(r"PASSWORD|PASS\b|PSK", re.I)


def main(args):
    lines = open(SDK, encoding="utf-8").read().split("\n")
    if args and args[0] == "--show":
        for name in args[1:]:
            assert not SECRET.search(name)
            hit = [l for l in lines if l.startswith(f"CONFIG_{name}=") or l == f"# CONFIG_{name} is not set"]
            print(hit[0] if hit else f"CONFIG_{name}: not present")
        return
    for a in args:
        name, val = a.split("=", 1)
        assert not SECRET.search(name), "refusing to touch credentials"
        new = f"# CONFIG_{name} is not set" if val == "n" else f"CONFIG_{name}={val}"
        idx = [i for i, l in enumerate(lines) if l.startswith(f"CONFIG_{name}=") or l == f"# CONFIG_{name} is not set"]
        if idx:
            for i in idx:
                lines[i] = new
        else:
            lines.append(new)
        print(new)
    open(SDK, "w", encoding="utf-8", newline="\n").write("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1:])
