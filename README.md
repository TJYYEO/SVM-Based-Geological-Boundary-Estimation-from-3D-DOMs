This repository contains Python workflows for estimating planar geological boundaries from three-dimensional spatial data using two complementary approaches: a linear Support Vector Machine (SVM) and orthogonal least-squares regression (LSR).

The SVM workflow estimates a separating plane from points assigned to two adjacent geological domains. It performs linear SVM classification, model evaluation, calculation of the decision-plane orientation, identification of support vectors, point-to-plane distance analysis, and visualisation of the results.

The LSR workflow fits a best-fit plane to points interpreted to lie directly on a geological boundary. It calculates point-to-plane residuals and structural orientation, applies a median absolute deviation-based filter to identify potential outliers, and refits the plane using the filtered dataset.

The two approaches are intended for complementary geological applications. The SVM workflow constrains a boundary from the spatial distribution of adjacent mapped geological domains, whereas the LSR workflow estimates a plane directly from mapped boundary observations. Both workflows report the resulting plane geometry, including dip and dip direction, and export numerical results and visualisations for further geological interpretation.

These scripts were developed to support reproducible geological-boundary analysis from three-dimensional digital outcrop and spatial mapping data.
