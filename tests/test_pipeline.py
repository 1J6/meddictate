import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
sys.argv = ["x"]
import meddictate as m


def run(name, phrases, expect):
    w = m.PhraseWriter(m.Target(), do_paste=False)
    for p in phrases:
        w.emit(p)
    print()
    ok = "OK " if w.doc == expect else "BAD"
    print(f"{ok} {name}: {w.doc!r}")
    if w.doc != expect:
        print(f"     expected {expect!r}")


run("new line = one enter",
    ["first line {new line} second line"],
    "First line\nSecond line")
run("new paragraph = two enters",
    ["first {period} {new paragraph} second"],
    "First.\n\nSecond")
run("new line new line stacks",
    ["one {new line} {new line} two"],
    "One\n\nTwo")
run("paragraph + line stacks (3 enters)",
    ["one {new paragraph} {new line} two"],
    "One\n\n\nTwo")
run("trailing newline kept across phrases",
    ["first line {new line}", "second line"],
    "First line\nSecond line")
run("leading newline phrase",
    ["first", "{new paragraph}", "second"],
    "First\n\nSecond")
run("section tag + dictated colon after paragraph",
    ["left {period}", "{new paragraph} [IMPRESSION] {colon} acute PE"],
    "Left.\n\nImpression: Acute PE")
run("section tag inline, no dictated punctuation",
    ["left {period}", "[IMPRESSION] acute PE"],
    "Left. Impression acute PE")
run("section tag first phrase",
    ["[EXAM TYPE] CT chest"],
    "Exam type CT chest")
run("backspace within phrase",
    ["hyperlipidemiaa backspace"],
    "Hyperlipidemia")
run("backspace stacks within phrase",
    ["hyperlipidemiaxx backspace backspace"],
    "Hyperlipidemia")
run("backspace reaches previous phrase",
    ["patient has GERDD", "backspace backspace"],
    "Patient has GER")
run("backspace then more text",
    ["patient has GERDD", "backspace {comma} and asthma"],
    "Patient has GERD, and asthma")
run("delete that within phrase",
    ["patient has diabetes delete that hypertension"],
    "Patient has hypertension")
run("delete that stacks",
    ["patient has type two diabetes {period} delete that delete that delete that asthma {period}"],
    "Patient has asthma.")
run("delete that across phrases",
    ["patient has diabetes {period}", "delete that hypertension {period}"],
    "Patient has hypertension.")
run("delete that undoes a paragraph",
    ["one {period} {new paragraph}", "delete that two"],
    "One. Two")
run("braced command token",
    ["diabetess {backspace} {comma} GERD"],
    "Diabetes, GERD")
run("capitalization follows deletions",
    ["patient is well {period} he", "delete that she is well"],
    "Patient is well. She is well")
run("filler + command",
    ["patient uh has uh backspace"],
    "Patient ha")
run("de-id tag fragments dropped, duplicate period collapsed (user sample)",
    ["[OVERALL IMPRESSION] is of a 75-year-old man who", "present for acute.", "e] examination.",
     "essionalName] {period}", "] {period}"],
    "Overall impression is of a 75-year-old man who present for acute examination.")
run("complete de-id tag dropped",
    ["seen by [ProfessionalName] on [Date] {period}"],
    "Seen by on.")
run("section tag survives the de-id sweep",
    ["left {period}", "[IMPRESSION] {colon} acute PE"],
    "Left. Impression: Acute PE")

