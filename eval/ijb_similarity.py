"""Paired IJB scoring report, comparing TAR at the same FAR constraint."""


def compare_tar_rows(cosine, dot):
    points = {}
    for far, cosine_point in cosine['points'].items():
        c = cosine_point['at_or_below_requested_far']
        d = dot['points'][far]['at_or_below_requested_far']
        points[far] = dict(
            cosine_tar_percent=c['tar_percent'],
            dot_tar_percent=d['tar_percent'],
            drop_percentage_points=c['tar_percent'] - d['tar_percent'],
            cosine_actual_far=c['actual_far'],
            dot_actual_far=d['actual_far'],
        )
    return points
