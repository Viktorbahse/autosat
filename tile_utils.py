import csv
import io
import math

import mercantile


def tiles_from_csv(path):
    """Read tiles from a line-delimited csv file.

    Args:
      file: the path to read the csv file from.

    Yields:
      The mercantile tiles from the csv file.
    """

    with open(path) as fp:
        reader = csv.reader(fp)

        for row in reader:
            if not row:
                continue

            yield mercantile.Tile(*map(int, row))

def fetch_image(session, url, timeout=10):
    """Fetches the image representation for a tile.

    Args:
      session: the HTTP session to fetch the image from.
      url: the tile imagery's url to fetch the image from.
      timeout: the HTTP timeout in seconds.

    Returns:
     The satellite imagery as bytes or None in case of error.
    """

    try:
        resp = session.get(url, timeout=timeout)
        resp.raise_for_status()
        return io.BytesIO(resp.content)
    except Exception:
        return None

def xyz_to_quadkey(x, y, z):
    x, y, z = int(x), int(y), int(z)
    quadkey = ''
    for i in range(z, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if (x & mask) != 0:
            digit += 1
        if (y & mask) != 0:
            digit += 2
        quadkey += str(digit)
    return quadkey

def latlon_to_yandex_tile(lat, lon, z):
    e = 0.0818191908426
    r = 2**(int(z) + 8) / 2
    b = lat * math.pi / 180
    p = (1 - e * math.sin(b)) / (1 + e * math.sin(b))
    t = math.tan(math.pi/4 + b/2) * p**(e/2)
    x = r * (1 + lon / 180)
    y = r * (1 - math.log(t) / math.pi)
    return [int(x/256), int(y/256)]

def pixel_to_location(tile, dx, dy):
    """Converts a pixel in a tile to a coordinate.

    Args:
      tile: the mercantile tile to calculate the location in.
      dx: the relative x offset in range [0, 1].
      dy: the relative y offset in range [0, 1].

    Returns:
      The coordinate for the pixel in the tile.
    """

    assert 0 <= dx <= 1, "x offset is in [0, 1]"
    assert 0 <= dy <= 1, "y offset is in [0, 1]"

    west, south, east, north = mercantile.bounds(tile)

    def lerp(a, b, c):
        return a + c * (b - a)

    lon = lerp(west, east, dx)
    lat = lerp(south, north, dy)

    return lon, lat