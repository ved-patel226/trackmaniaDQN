from pathlib import Path
import numpy as np
from pygbx import Gbx, GbxType
from scipy.interpolate import make_interp_spline


def gbx_to_raw_pos_list(gbx_path: Path):
    """
    Read a .gbx file, extract the raw positions of the best ghost included in that file.
    """
    gbx = Gbx(str(gbx_path))
    ghosts = gbx.get_classes_by_ids([GbxType.CTN_GHOST])
    assert len(ghosts) > 0, "The file does not contain any ghost."
    ghost = min(ghosts, key=lambda g: g.cp_times[-1])
    if ghost.num_respawns != 0:
        print("")
        print("------------    Warning: The ghost contains respawns  ---------------")
        print("")
    records_to_keep = round(ghost.race_time / 100)

    print(
        ghost.race_time,
        f"ghost has {len(ghost.records)} records and {len(ghost.control_entries)} control entries",
    )
    print(
        "Keeping",
        records_to_keep,
        "out of",
        len(ghost.records),
        "records for a race time of",
        ghost.race_time / 1000,
    )

    raw_positions_list = []
    for r in ghost.records[:records_to_keep]:
        raw_positions_list.append(np.array([r.position.x, r.position.y, r.position.z]))

    return raw_positions_list


def extract_cp_distance_interval(
    raw_position_list: list, target_distance_between_cp_m: float
):
    """
    :param raw_position_list:               a list of 3D coordinates.

    This function saves on disk a 2D numpy array of shape (N, 3) with the following properties.
    - The first point of the array is raw_position_list[0]
    - The middle of the last and second to last points of the array is raw_position_list[-1]
    - All points in the 2D array are distant of approximately target_distance_between_cp_m from their neighbours.
    - All points of the array lie on the path defined by raw_position_list

    In short, this function resamples a path given in input to return regularly spaced checkpoints.

    It is highly likely that there exists a one-liner in numpy to do all this, but I have yet to find it...
    """
    interpolation_function = make_interp_spline(
        x=range(len(raw_position_list)), y=raw_position_list, k=1
    )
    raw_position_list = interpolation_function(
        np.arange(0, len(raw_position_list) - 1 + 1e-6, 0.01)
    )
    a = np.array(raw_position_list)
    b = np.linalg.norm(
        a[:-1] - a[1:], axis=1
    )  # b[i] : distance traveled between point i and point i+1, for i > 0
    c = np.pad(
        b.cumsum(), (1, 0)
    )  # c[i] : distance traveled between point 0 and point i
    number_zones = (
        round(c[-1] / target_distance_between_cp_m - 0.5) + 0.5
    )  # half a zone for the end
    zone_length = c[-1] / number_zones
    index_first_pos_in_new_zone = np.unique(c // zone_length, return_index=True)[1][1:]
    index_last_pos_in_current_zone = index_first_pos_in_new_zone - 1
    w1 = 1 - (c[index_last_pos_in_current_zone] % zone_length) / zone_length
    w2 = (c[index_first_pos_in_new_zone] % zone_length) / zone_length
    zone_centers = a[index_last_pos_in_current_zone] + (
        a[index_first_pos_in_new_zone] - a[index_last_pos_in_current_zone]
    ) * (w1 / (1e-4 + w1 + w2)).reshape((-1, 1))
    zone_centers = np.vstack(
        (
            raw_position_list[0][None, :],
            zone_centers,
            (2 * raw_position_list[-1] - zone_centers[-1])[None, :],
        )
    )
    np.save("../map.npy", np.array(zone_centers).round(4))
    return zone_centers


def find_nearest_checkpoint(position, zone_centers):
    min_distance = float("inf")
    min_idx = -1

    for i, zone_center in enumerate(zone_centers):
        if distance_between(position, zone_center) < min_distance:
            min_distance = distance_between(position, zone_center)
            min_idx = i

    return min_distance, min_idx


def distance_between(p1, p2):
    """
    Calculate the Euclidean distance between two 3D points.
    Both points should be array-like with three elements.
    """
    return np.linalg.norm(np.array(p1) - np.array(p2))


def main() -> None:
    path = Path(
        r"C:\Users\talk2_6h7jpbd\Documents\TrackMania\Tracks\Replays\AI #5_ved-patel226(00'26''50).Replay.Gbx"
    )

    base_dir = Path(r"C:\Users\talk2_6h7jpbd\Documents\TrackMania")

    raw_positions_list = gbx_to_raw_pos_list(path)
    raw_positions_list = np.array(raw_positions_list)
    np.save("map2.npy", raw_positions_list)

    # zone_centers = extract_cp_distance_interval(raw_positions_list, 0.1)

    # print(find_nearest_checkpoint(np.array([0, 0, 0]), zone_centers))


if __name__ == "__main__":
    main()
