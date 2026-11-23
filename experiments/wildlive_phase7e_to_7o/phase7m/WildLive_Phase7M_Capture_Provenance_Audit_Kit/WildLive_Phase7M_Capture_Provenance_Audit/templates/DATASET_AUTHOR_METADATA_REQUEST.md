# Draft correspondence — WildLive flight / capture-event provenance

**Suggested subject:** Request for WildLive video lineage and capture-event metadata (elephant sequence validation)

Dear WildLive dataset team,

I am working on elephant trajectory forecasting using the WildLive dataset (DOI: 10.5523/bris.ewnuoroebuae20vei0cu6zu69). Thank you for making the annotations and images available for research.

For a scientifically defensible recording-level evaluation, I would be grateful for clarification of the source capture events and temporal sampling for these four folders:

- `DJI_0207`
- `DJI_0204_video1`
- `DJI_0204_video2`
- `DJI_0161_video1`

We have found that `DJI_0204_video1` and `DJI_0204_video2` share substantial visual background content. We do not want to assume they are independent drone flights, or infer temporal separation from the folder names. Could you provide, if available:

1. A **non-sensitive source flight/session identifier** for each sequence and a mapping from each exported `videoN` segment to its original camera video file. In particular, are the two `DJI_0204` sequences cut from one source recording, from different flights at the same site, or from independent missions?
2. Whether `DJI_0207` and `DJI_0161_video1` share any source flight/session, capture time, or herd encounter with the `DJI_0204` sequences or with one another.
3. The true **capture-frame rate** (not onboard inference throughput), any temporal subsampling/annotation stride, and whether frame indices 0, 1, 2, ... correspond to consecutive captured frames.
4. Original clip start/end timestamps with time zone, or relative offsets in their parent videos; **approximate dates or anonymized session IDs are sufficient** if precise time or location metadata is sensitive.
5. Any public documentation/table describing the relationships among all 22 released sequence folders; ideally the parent capture ID, sequence offset and species labels per video.
6. If available, camera-motion/telemetry information (camera orientation or altitude over time), or a note confirming that such data is not provided.

For elephant conservation and privacy reasons, **we do not need precise GPS coordinates or sensitive wildlife locations**. A compact anonymized flight-ID/clip-lineage table would already be very helpful.

We will cite both the WildLive dataset and its associated publication and follow the published dataset licence. Thank you for your assistance.

Best regards,
[Researcher name / university / contact]

---

**Important:** This is a draft only. It has NOT been sent. Do not invent or assume a recipient email address. The public dataset landing page and University of Bristol project/research pages provide routes for finding the appropriate author or official contact.

Dataset: https://data.bristol.ac.uk/data/dataset/ewnuoroebuae20vei0cu6zu69

Project: https://dat-nguyenvn.github.io/WildLive/
