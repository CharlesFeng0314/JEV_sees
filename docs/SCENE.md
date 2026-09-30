**English** | [中文](SCENE.zh.md)

# What JEV receives

JEV does not receive pixels. `Sees.ask` builds an internal `given_that` from the user sentence, the objects visible now, the objects remembered from earlier frames, and the uncertainties. The description stays inside the SDK. It is not copied onto the response.

Each object fact keeps `object_id`, label, confidence, `bbox_xyxy`, and a pixel centroid. A depth frame also adds a camera-frame `position_m` and, when two objects are near, `gap_m`. Masks, full score vectors, and the pose trail stay out of the prompt. Color is the one attribute that is kept.

## Relations

Geometric facts are computed in code. JEV decides what they mean.

A pair is recorded when the boxes overlap or the box gap is within 40 pixels. The fact includes `bbox_iou`, `box_gap_px`, `centroid_distance_px`, and, once both objects have a previous centroid, `approaching`. Distant pairs are omitted, and if the prompt is still too long they are dropped first.

## Memory

Object identity is stable across frames. The same `object_id` is merged rather than replaced. Labels are a majority vote, the latest box is what the prompt sees, and an object goes stale after three missed frames.

At most 16 objects are kept from memory and 16 from the current frame, preferring objects that are visible, not stale, and seen more often. If that is still too long, lower-priority objects are dropped down to 4.

## Budget

JEV's input limit is 31,000 tokens. The same budget used by the robot project applies here: the target is 24,000 characters, and the character count is the conservative token estimate. The budget numbers stay inside `given_that`.

## Why the RGB path is not boxes alone

`perceive_rgb` is the product path for a color frame: YOLOv8-World boxes, a CLIP color name, then a body-color heuristic. On `assets/bus.jpg` the heuristic says the bus is blue and CLIP says green, so the color written into `given_that` is blue. A chromatic heuristic wins. CLIP remains the name source for depth clusters.

`perceive_rgbd` lets depth clusters decide how many objects exist. CLIP names each cluster. YOLO-World is attached only when a box agrees with a cluster. On wrist frame 13 the detector named nothing at 0.25 and at 0.02, while the pipeline still returned spoon, cube, water bottle, and cracker box.

MobileSAM, on the same bus image, returned 83 masks and no category names in about 1.7 s. Masks without names cannot answer a closed question, so it is not on the default path.

The timed comparison is [`pipeline_benchmark.json`](pipeline_benchmark.json). It is one development run on two public images and three wrist frames, not a COCO mAP.
