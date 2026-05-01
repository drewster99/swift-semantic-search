import Foundation

/// Built-in demo corpus shared between cli-demo and gui-demo. Twenty-four short
/// sentences across six everyday topics — chosen so semantic queries that
/// don't share keywords with the document text still rank the right one on top.
enum DemoCorpus {
    struct Document: Identifiable {
        let id: String
        let topic: String
        let text: String
    }

    static let blurb: String = """
        24 short sentences across six topics — cooking, software, astronomy,
        music, fitness, gardening — chosen so semantic queries that don't share
        words with the documents still rank the right one on top.
        """

    static let documents: [Document] = [
        Document(id: "cook-risotto",    topic: "cooking",   text: "To make a great risotto, stir arborio rice constantly and add warm stock one ladle at a time until the grains turn creamy and al dente."),
        Document(id: "cook-sourdough",  topic: "cooking",   text: "A healthy sourdough starter is maintained at roughly equal parts flour, water, and mature leaven by weight, and fed once or twice a day."),
        Document(id: "cook-knife",      topic: "cooking",   text: "To sharpen a dull kitchen knife on a whetstone, keep the blade at a consistent fifteen- to twenty-degree angle and alternate strokes on each side."),
        Document(id: "cook-bbq",        topic: "cooking",   text: "Slow-smoked barbecue pork shoulder cooks for eight to twelve hours around 225 degrees Fahrenheit over indirect heat until the bark is dark and the meat pulls apart easily."),

        Document(id: "sw-actor",        topic: "software",  text: "Swift actors serialize access to their mutable state so concurrent code running on multiple threads cannot introduce low-level data races."),
        Document(id: "sw-venv",         topic: "software",  text: "A Python virtual environment isolates a single project's installed packages so their versions cannot collide with any other project on the same machine."),
        Document(id: "sw-rebase",       topic: "software",  text: "Interactive rebase in Git lets a developer squash, reorder, edit, or drop individual commits to clean up history before a feature branch is merged."),
        Document(id: "sw-index",        topic: "software",  text: "A well-chosen database index turns a full table scan into a logarithmic lookup, dramatically speeding up queries that filter on the indexed columns."),

        Document(id: "astro-blackhole", topic: "astronomy", text: "When a very massive star runs out of fuel, its core collapses under its own gravity and forms a black hole from which not even light can escape."),
        Document(id: "astro-exoplanet", topic: "astronomy", text: "Astronomers detect exoplanets by measuring the tiny dip in a distant star's brightness when an orbiting planet transits across its face."),
        Document(id: "astro-aurora",    topic: "astronomy", text: "The aurora borealis appears when charged particles from the solar wind collide with oxygen and nitrogen atoms high in Earth's upper atmosphere."),
        Document(id: "astro-comet",     topic: "astronomy", text: "A comet is an icy body that grows a long glowing tail as it swings near the Sun and its frozen volatiles sublimate into gas and dust."),

        Document(id: "music-key",       topic: "music",     text: "A key signature written at the start of a staff lists the sharps or flats that apply to every matching note throughout the piece."),
        Document(id: "music-iiVI",      topic: "music",     text: "The ii-V-I progression is the backbone of jazz harmony, cycling through the supertonic, dominant, and tonic chords to resolve strongly back home."),
        Document(id: "music-dorian",    topic: "music",     text: "The Dorian mode is a minor scale with a raised sixth degree, giving it a bright, folk-like flavor that distinguishes it from natural minor."),
        Document(id: "music-fifth",     topic: "music",     text: "A perfect fifth is the consonant musical interval spanning seven semitones, such as the distance from C up to G on a piano keyboard."),

        Document(id: "fit-cadence",     topic: "fitness",   text: "A running cadence near one hundred eighty steps per minute shortens ground contact time, smooths out stride mechanics, and tends to reduce overuse injuries."),
        Document(id: "fit-vo2",         topic: "fitness",   text: "VO2 max improves fastest with repeated four- to six-minute intervals performed at near-maximal effort, separated by easy recovery jogs."),
        Document(id: "fit-zone2",       topic: "fitness",   text: "Zone 2 training at a conversational heart rate builds a deep aerobic base over many weeks without piling up fatigue or injury risk."),
        Document(id: "fit-long-run",    topic: "fitness",   text: "A weekly long run for marathon preparation should increase in distance gradually, typically adding no more than about ten percent from week to week."),

        Document(id: "garden-compost",  topic: "gardening", text: "A balanced compost pile mixes roughly three parts carbon-rich browns like dry leaves with one part nitrogen-rich greens like fresh grass clippings and kitchen scraps."),
        Document(id: "garden-prune",    topic: "gardening", text: "Pruning fruit trees during late winter dormancy encourages vigorous spring regrowth and a heavier, better-shaped harvest the following summer."),
        Document(id: "garden-mulch",    topic: "gardening", text: "A two-inch layer of mulch spread over garden beds conserves soil moisture, insulates roots from temperature swings, and suppresses weed germination."),
        Document(id: "garden-tomato",   topic: "gardening", text: "Gardeners often plant tomatoes alongside basil in the belief that the herb improves tomato flavor and helps repel common pests like aphids and hornworms.")
    ]
}
