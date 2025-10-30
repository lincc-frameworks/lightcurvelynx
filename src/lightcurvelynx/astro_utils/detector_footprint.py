"""A wrapper class for representing detector footprints (which are stored as
astropy PixelRegion). This class provides methods for checking if points are within
the footprint and for plotting the footprint."""

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from regions import PixCoord, RectanglePixelRegion, RectangleSkyRegion, SkyRegion


class DetectorFootprint:
    """A wrapper class for representing detector footprints.

    Attributes
    ----------
    region : astropy.regions.PixelRegion
        The astropy PixelRegion representing the footprint.
    wcs : astropy.wcs.WCS or None
        The WCS associated with the region, if any.

    Parameters
    ----------
    region : astropy.regions.SkyRegion or astropy.regions.PixelRegion
        The astropy SkyRegion or PixelRegion representing the footprint. SkyRegions
        will be converted to PixelRegions using the provided WCS or a default WCS.
    wcs : astropy.wcs.WCS or None
        The WCS associated with the region, if any.
    pixel_scale : float or None
        The pixel scale in arcseconds/pixel, this is required if no WCS is provided.
    center_pixels : tuple of float, optional
        The pixel coordinates of the center of the detector. Default is (0.5, 0.5) for
        the center of the (0, 0) pixel. This is only used if no WCS is provided and
        a default WCS is created.
    """

    def __init__(self, region, *, wcs=None, pixel_scale=None, center_pixels=(0.5, 0.5), **kwargs):
        # Create a default WCS if none is provided.
        if wcs is None:
            if pixel_scale is None:
                raise ValueError("Either wcs or pixel_scale must be provided.")
            if pixel_scale <= 0:
                raise ValueError("pixel_scale must be positive.")
            pixel_scale_deg = pixel_scale / 3600.0  # Convert to degrees/pixel

            # Create a simple TAN WCS centered on (0.0, 0.0) with the given pixel scale.
            wcs = WCS(naxis=2)
            wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
            wcs.wcs.crval = [0.0, 0.0]  # Centered on RA=0.0, dec=0.0
            wcs.wcs.crpix = [center_pixels[0], center_pixels[1]]
            wcs.wcs.cdelt = [-pixel_scale_deg, pixel_scale_deg]  # The given pixel scale in degrees/pixel
        self.wcs = wcs

        # Store the region as a pixel region, since we will always need to do a conversion
        # as part of the contains method otherwise.
        self.region = region
        if isinstance(self.region, SkyRegion):
            self.region = self.region.to_pixel(self.wcs)

        # The bounding box must contain the origin (0,0) since the footprint is defined
        # relative to the center of the detector.
        bbox = self.region.bounding_box
        if not (bbox.ixmin <= 0 <= bbox.ixmax and bbox.iymin <= 0 <= bbox.iymax):
            raise ValueError("The bounding box of the region must contain the origin (0,0).")

    @classmethod
    def from_sky_rect(cls, width, height, wcs=None, pixel_scale=None, **kwargs):
        """Create a rectangular footprint in degrees.

        Parameters
        ----------
        width : float
            Width of the rectangle in degrees.
        height : float
            Height of the rectangle in degrees.
        wcs : astropy.wcs.WCS, optional
            The WCS associated with the region. If None, a default WCS will be created.
        pixel_scale : float, optional
            The pixel scale in arcseconds/pixel, this is required if no WCS is provided.
        **kwargs : dict
            Additional keyword arguments to pass to the RectangleSkyRegion constructor.

        Returns
        -------
        DetectorFootprint
            The rectangular detector footprint.
        """
        center = SkyCoord(ra=0.0, dec=0.0, unit="deg", frame="icrs")
        region = RectangleSkyRegion(
            center=center,
            width=width * u.deg,
            height=height * u.deg,
            angle=0.0 * u.deg,
            **kwargs,
        )
        return cls(region, wcs=wcs, pixel_scale=pixel_scale)

    @classmethod
    def from_pixel_rect(cls, width, height, wcs=None, pixel_scale=None, **kwargs):
        """Create a rectangular footprint in pixels.

        Parameters
        ----------
        width : float
            Width of the rectangle in pixels.
        height : float
            Height of the rectangle in pixels.
        wcs : astropy.wcs.WCS, optional
            The WCS associated with the region. If None, a default WCS will be created.
        pixel_scale : float, optional
            The pixel scale in arcseconds/pixel, this is required if no WCS is provided.
        **kwargs : dict
            Additional keyword arguments to pass to the RectangleSkyRegion constructor.

        Returns
        -------
        DetectorFootprint
            The rectangular detector footprint.
        """
        center = PixCoord(x=0.0, y=0.0)
        region = RectanglePixelRegion(
            center=center,
            width=width,
            height=height,
            angle=0.0 * u.deg,
            **kwargs,
        )
        return cls(region, wcs=wcs, pixel_scale=pixel_scale)

    @staticmethod
    def rotate_to_center(ra, dec, center_ra, center_dec, *, rotation=None):
        """Transform the given points, represented by (ra, dec) in degrees,
        to a local coordinate system centered on (center_ra, center_dec), accounting
        for rotation if specified.

        Note
        ----
        This method is vectorized so it can handle an array of points and an (equally
        sized) array of center points. The transformation is done on each point for the
        corresponding center point.

        Parameters
        ----------
        ra : np.ndarray
            Right ascension in degrees.
        dec : np.ndarray
            Declination in degrees.
        center_ra : np.ndarray
            Center right ascension of the detector in degrees.
        center_dec : np.ndarray
            Center declination of the detector in degrees.
        rotation : np.ndarray, optional
            The rotation angle of the detector for each pointing in degrees clockwise.
            Used to represent non-axis-aligned footprints.

        Returns
        -------
        tuple of np.ndarray
            Transformed (lon, lat) offsets from the center in degrees.
        """
        if rotation is None:
            rotation = np.zeros_like(ra)

        target = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
        origin = SkyCoord(ra=center_ra * u.deg, dec=center_dec * u.deg, frame="icrs")
        offset_frame = origin.skyoffset_frame(rotation=rotation * u.deg)
        offset = target.transform_to(offset_frame)

        # Center everything on 0.0
        lon_t = offset.lon.deg
        lon_t[lon_t > 180.0] -= 360.0

        lat_t = offset.lat.deg
        lat_t[lat_t > 90.0] -= 180.0

        return lon_t, lat_t

    def sky_to_pixel(self, ra, dec, center_ra, center_dec, *, rotation=None):
        """Transform sky coordinates (ra, dec) to pixel coordinates.

        Parameters
        ----------
        ra : float or array-like
            Right ascension in degrees.
        dec : float or array-like
            Declination in degrees.
        center_ra : float
            Center right ascension of the detector in degrees.
        center_dec : float
            Center declination of the detector in degrees.
        rotation : float or array-like, optional
            The rotation angle of the detector for each pointing in degrees clockwise.
            Used to represent non-axis-aligned footprints.

        Returns
        -------
        tuple of np.ndarray
            Pixel coordinates (x, y) corresponding to the input sky coordinates.
        """
        lon_t, lat_t = self.rotate_to_center(ra, dec, center_ra, center_dec, rotation=rotation)
        sky_pts = SkyCoord(lon_t, lat_t, unit="deg")
        pixel_x, pixel_y = self.wcs.world_to_pixel(sky_pts)
        return pixel_x, pixel_y

    def contains(self, ra, dec, center_ra, center_dec, *, rotation=None):
        """Check that given points, represented by (ra, dec) in degrees,
        are within the footprint.

        Parameters
        ----------
        ra : float or array-like
            Right ascension in degrees.
        dec : float or array-like
            Declination in degrees.
        center_ra : float or array-like
            Center right ascension of the detector in degrees.
        center_dec : float or array-like
            Center declination of the detector in degrees.
        rotation : np.ndarray, optional
            The rotation angle of the detector for each pointing in degrees clockwise.
            Used to represent non-axis-aligned footprints.

        Returns
        -------
        bool or array-like of bool
            True if the point is within the footprint, False otherwise.
        """
        scalar_data = np.isscalar(ra)

        # Make all inputs into arrays and confirm they have the same shape.
        ra = np.atleast_1d(ra)
        dec = np.atleast_1d(dec)
        if ra.shape != dec.shape:
            raise ValueError("ra and dec must have the same shape.")

        center_ra = np.atleast_1d(center_ra)
        center_dec = np.atleast_1d(center_dec)
        if center_ra.shape != ra.shape:
            if center_ra.shape != (1,):
                raise ValueError("center_ra must have the same shape as ra and dec.")
            else:
                center_ra = np.full(ra.shape, center_ra[0])
        if center_dec.shape != dec.shape:
            if center_dec.shape != (1,):
                raise ValueError("center_dec must have the same shape as ra and dec.")
            else:
                center_dec = np.full(dec.shape, center_dec[0])

        # Rotation is optional, but if provided, it must match the shape of ra and dec.
        if rotation is not None:
            rotation = np.atleast_1d(rotation)
            if rotation.shape != ra.shape:
                if rotation.shape != (1,):
                    raise ValueError("rotation must have the same shape as ra and dec.")
                else:
                    rotation = np.full(ra.shape, rotation[0])

        # Rotate the points to be relative to the center and convert to pixel coordinates.
        pixel_x, pixel_y = self.sky_to_pixel(ra, dec, center_ra, center_dec, rotation=rotation)
        pix_coord = PixCoord(x=pixel_x, y=pixel_y)

        # Check if each point is within the footprint.
        result = self.region.contains(pix_coord)
        if scalar_data:
            return result[0]
        return result

    def compute_radius(self):
        """Compute an approximate bounding radius of the footprint in degrees.

        Note
        ----
        This radius is based on the bounding box of the footprint (with a small
        pixel padding), so it will be larger than the actual radius of the footprint.

        Returns
        -------
        float
            The radius of the footprint in degrees.
        """
        # Compute the corners of the bounding box (using a pixel buffer to handle different
        # centering schemes) and convert to sky coordinates.
        bbox = self.region.bounding_box
        corner_pix_x = np.array([bbox.ixmin - 1.0, bbox.ixmin - 1.0, bbox.ixmax + 1.0, bbox.ixmax + 1.0])
        corner_pix_y = np.array([bbox.iymin - 1.0, bbox.iymax + 1.0, bbox.iymax + 1.0, bbox.iymin - 1.0])
        corner_ra, corner_dec = self.wcs.pixel_to_world_values(corner_pix_x, corner_pix_y)
        sky_corners = SkyCoord(ra=corner_ra, dec=corner_dec, unit="deg", frame="icrs")

        center = SkyCoord(ra=0.0, dec=0.0, unit="deg", frame="icrs")
        separations = center.separation(sky_corners)
        return np.max(separations.deg)

    def compute_sky_region(self, center_ra, center_dec, *, rotation=0.0):
        """Compute the sky region of the footprint at the given center position.

        Parameters
        ----------
        center_ra : float
            Center right ascension of the detector in degrees.
        center_dec : float
            Center declination of the detector in degrees.
        rotation : float, optional
            The rotation angle of the detector for each pointing in degrees clockwise.
            Used to represent non-axis-aligned footprints. Default is 0.0.

        Returns
        -------
        astropy.regions.SkyRegion
            The sky region representing the footprint at the given center position.
        astropy.wcs.WCS
            The WCS representing the footprint at the given center position.
        """
        # Perform the rotation in pixel coordinates first.
        if rotation != 0.0:
            center = PixCoord(x=0.0, y=0.0)
            pixel_region = self.region.rotate(center, rotation * u.deg)
        else:
            pixel_region = self.region

        # Convert to sky region and set the center.
        sky_region = pixel_region.to_sky(self.wcs)
        sky_region.center = SkyCoord(ra=center_ra, dec=center_dec, unit="deg", frame="icrs")

        # Create a copy of the WCS with the new center.
        sky_wcs = self.wcs.deepcopy()
        sky_wcs.wcs.crval = [center_ra, center_dec]

        return sky_region, sky_wcs

    def plot(
        self,
        *,
        ax=None,
        figure=None,
        center_ra=0,
        center_dec=0,
        rotation=0,
        point_ra=None,
        point_dec=None,
        **kwargs,
    ):
        """Plot the footprint using matplotlib and an optional set of points to overlay.

        Parameters
        ----------
        ax : matplotlib.pyplot.Axes or None, optional
            Axes, If None, a new axes will be created. None by default.
        figure : matplotlib.pyplot.Figure or None
            Figure, If None, a new figure will be created. None by default.
        center_ra : float, optional
            Center right ascension of the detector in degrees. Default is 0.
        center_dec : float, optional
            Center declination of the detector in degrees. Default is 0.
        rotation : np.ndarray, optional
            The rotation angle of the detector for each pointing in degrees clockwise.
            Used to represent non-axis-aligned footprints.
            Default is 0.
        point_ra : array-like or None, optional
            Right ascension of points to overlay in degrees. None by default.
        point_dec : array-like or None, optional
            Declination of points to overlay in degrees. None by default.
        **kwargs : dict
            Optional parameters to pass to the plotting function
        """
        if ax is None:
            if figure is None:
                figure = plt.figure()
            ax = figure.add_axes([0, 0, 1, 1])

        # Plot the bounds of the footprint.
        artist = self.region.as_artist()
        ax.add_artist(artist)

        # Get the bounding box in pixel coordinates. Expand out to ensure the full
        # footprint is visible.
        bbox = self.region.bounding_box
        width = bbox.ixmax - bbox.ixmin
        height = bbox.iymax - bbox.iymin
        xmin = bbox.ixmin - 0.5 * width
        xmax = bbox.ixmax + 0.5 * width
        ymin = bbox.iymin - 0.5 * height
        ymax = bbox.iymax + 0.5 * height

        # If points are provided, overlay them on the plot.
        if point_ra is not None and point_dec is not None:
            # Compute the transformed pixel coordinates relative to the center.
            pixel_x, pixel_y = self.sky_to_pixel(
                point_ra, point_dec, center_ra, center_dec, rotation=rotation
            )

            # Since we have already transformed the points to be relative to the center,
            # we can check if they are within the footprint without further transformation.
            isin = self.region.contains(PixCoord(x=pixel_x, y=pixel_y))

            ax.scatter(pixel_x[~isin], pixel_y[~isin], color="red", marker="x", label="Outside Footprint")
            ax.scatter(pixel_x[isin], pixel_y[isin], color="green", marker="o", label="Inside Footprint")
            ax.legend()

            # Adjust the bounding box to include all the points.
            if xmin < pixel_x.min():
                xmin = pixel_x.min() - 2.0
            if xmax > pixel_x.max():
                xmax = pixel_x.max() + 2.0
            if ymin < pixel_y.min():
                ymin = pixel_y.min() - 2.0
            if ymax > pixel_y.max():
                ymax = pixel_y.max() + 2.0

        ax.set_title("Detector Footprint")
        ax.set_xlabel("Pixel X")
        ax.set_ylabel("Pixel Y")
        ax.set_aspect("equal")
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        plt.show()
