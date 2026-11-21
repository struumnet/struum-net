//! A small dependency-free path tracer: spheres with diffuse, metal and glass
//! materials under a sky light, with depth of field.
//!
//! Every pixel seeds its own RNG from its global index, so the image is the same
//! however the work is split across workers and cores. Rendering is very
//! compute-heavy per byte sent (a few config words in, one u32 per pixel out),
//! which is the case where adding more devices pays off.

use std::ops::{Add, Mul, Neg, Sub};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct V(pub f32, pub f32, pub f32);

impl Add for V {
    type Output = V;
    fn add(self, o: V) -> V {
        V(self.0 + o.0, self.1 + o.1, self.2 + o.2)
    }
}
impl Sub for V {
    type Output = V;
    fn sub(self, o: V) -> V {
        V(self.0 - o.0, self.1 - o.1, self.2 - o.2)
    }
}
impl Mul<f32> for V {
    type Output = V;
    fn mul(self, s: f32) -> V {
        V(self.0 * s, self.1 * s, self.2 * s)
    }
}
impl Mul for V {
    type Output = V;
    fn mul(self, o: V) -> V {
        V(self.0 * o.0, self.1 * o.1, self.2 * o.2)
    }
}
impl Neg for V {
    type Output = V;
    fn neg(self) -> V {
        V(-self.0, -self.1, -self.2)
    }
}

impl V {
    fn dot(self, o: V) -> f32 {
        self.0 * o.0 + self.1 * o.1 + self.2 * o.2
    }
    fn cross(self, o: V) -> V {
        V(self.1 * o.2 - self.2 * o.1, self.2 * o.0 - self.0 * o.2, self.0 * o.1 - self.1 * o.0)
    }
    fn len(self) -> f32 {
        self.dot(self).sqrt()
    }
    fn unit(self) -> V {
        self * (1.0 / self.len())
    }
    fn near_zero(self) -> bool {
        self.0.abs() < 1e-6 && self.1.abs() < 1e-6 && self.2.abs() < 1e-6
    }
}

fn hash(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^ (x >> 16)
}

/// xorshift32, seeded through a hash so neighbouring seeds are uncorrelated.
pub struct Rng(u32);

impl Rng {
    pub fn new(seed: u32) -> Self {
        Rng(hash(seed) | 1)
    }
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        (self.0 >> 8) as f32 / 16_777_216.0
    }
    fn in_unit_sphere(&mut self) -> V {
        loop {
            let p = V(self.next() * 2.0 - 1.0, self.next() * 2.0 - 1.0, self.next() * 2.0 - 1.0);
            if p.dot(p) < 1.0 {
                return p;
            }
        }
    }
    fn in_unit_disk(&mut self) -> V {
        loop {
            let p = V(self.next() * 2.0 - 1.0, self.next() * 2.0 - 1.0, 0.0);
            if p.dot(p) < 1.0 {
                return p;
            }
        }
    }
}

#[derive(Clone, Copy)]
enum Material {
    Diffuse(V),
    Metal(V, f32),
    Glass(f32),
}

struct Sphere {
    center: V,
    radius: f32,
    material: Material,
}

struct Camera {
    origin: V,
    lower_left: V,
    horizontal: V,
    vertical: V,
    u: V,
    v: V,
    lens_radius: f32,
}

impl Camera {
    fn new(from: V, at: V, vfov_deg: f32, aspect: f32, aperture: f32, focus: f32) -> Self {
        let h = (vfov_deg.to_radians() / 2.0).tan();
        let height = 2.0 * h;
        let width = aspect * height;
        let w = (from - at).unit();
        let u = V(0.0, 1.0, 0.0).cross(w).unit();
        let v = w.cross(u);
        let horizontal = u * (width * focus);
        let vertical = v * (height * focus);
        Self {
            origin: from,
            lower_left: from - horizontal * 0.5 - vertical * 0.5 - w * focus,
            horizontal,
            vertical,
            u,
            v,
            lens_radius: aperture / 2.0,
        }
    }

    fn ray(&self, s: f32, t: f32, rng: &mut Rng) -> (V, V) {
        let rd = rng.in_unit_disk() * self.lens_radius;
        let offset = self.u * rd.0 + self.v * rd.1;
        let origin = self.origin + offset;
        (origin, self.lower_left + self.horizontal * s + self.vertical * t - origin)
    }
}

pub struct Scene {
    spheres: Vec<Sphere>,
    camera: Camera,
}

struct Hit {
    t: f32,
    point: V,
    normal: V,
    front: bool,
    material: Material,
}

impl Scene {
    /// A field of small random spheres around three large ones. `seed` picks the layout.
    pub fn new(seed: u32, aspect: f32) -> Self {
        let mut rng = Rng::new(seed);
        let mut spheres = vec![
            Sphere { center: V(0.0, -1000.0, 0.0), radius: 1000.0, material: Material::Diffuse(V(0.5, 0.5, 0.5)) },
            Sphere { center: V(0.0, 1.0, 0.0), radius: 1.0, material: Material::Glass(1.5) },
            Sphere { center: V(-4.0, 1.0, 0.0), radius: 1.0, material: Material::Diffuse(V(0.4, 0.2, 0.1)) },
            Sphere { center: V(4.0, 1.0, 0.0), radius: 1.0, material: Material::Metal(V(0.7, 0.6, 0.5), 0.0) },
        ];
        for a in -6..6 {
            for b in -6..6 {
                let center = V(a as f32 + 0.9 * rng.next(), 0.2, b as f32 + 0.9 * rng.next());
                if (center - V(4.0, 0.2, 0.0)).len() < 0.9 || (center - V(-4.0, 0.2, 0.0)).len() < 0.9 {
                    continue;
                }
                let pick = rng.next();
                let material = if pick < 0.75 {
                    let c1 = V(rng.next(), rng.next(), rng.next());
                    let c2 = V(rng.next(), rng.next(), rng.next());
                    Material::Diffuse(c1 * c2)
                } else if pick < 0.92 {
                    let c = V(0.5 + rng.next() * 0.5, 0.5 + rng.next() * 0.5, 0.5 + rng.next() * 0.5);
                    Material::Metal(c, rng.next() * 0.5)
                } else {
                    Material::Glass(1.5)
                };
                spheres.push(Sphere { center, radius: 0.2, material });
            }
        }
        let camera = Camera::new(V(13.0, 2.0, 3.0), V(0.0, 0.0, 0.0), 20.0, aspect, 0.1, 10.0);
        Self { spheres, camera }
    }

    fn hit(&self, origin: V, dir: V, t_min: f32, mut t_max: f32) -> Option<Hit> {
        let mut best: Option<(f32, &Sphere)> = None;
        let a = dir.dot(dir);
        for s in &self.spheres {
            let oc = origin - s.center;
            let half_b = oc.dot(dir);
            let c = oc.dot(oc) - s.radius * s.radius;
            let disc = half_b * half_b - a * c;
            if disc < 0.0 {
                continue;
            }
            let sq = disc.sqrt();
            let mut t = (-half_b - sq) / a;
            if t < t_min || t > t_max {
                t = (-half_b + sq) / a;
                if t < t_min || t > t_max {
                    continue;
                }
            }
            t_max = t;
            best = Some((t, s));
        }
        best.map(|(t, s)| {
            let point = origin + dir * t;
            let outward = (point - s.center) * (1.0 / s.radius);
            let front = dir.dot(outward) < 0.0;
            Hit { t, point, normal: if front { outward } else { -outward }, front, material: s.material }
        })
    }

    /// Radiance along a ray, following up to `depth` bounces.
    fn trace(&self, mut origin: V, mut dir: V, depth: u32, rng: &mut Rng) -> V {
        let mut attenuation = V(1.0, 1.0, 1.0);
        for _ in 0..depth {
            let Some(hit) = self.hit(origin, dir, 1e-3, f32::INFINITY) else {
                let t = 0.5 * (dir.unit().1 + 1.0);
                let sky = V(1.0, 1.0, 1.0) * (1.0 - t) + V(0.5, 0.7, 1.0) * t;
                return attenuation * sky;
            };
            debug_assert!(hit.t > 0.0);
            let (albedo, next) = match hit.material {
                Material::Diffuse(albedo) => {
                    let mut d = hit.normal + rng.in_unit_sphere().unit();
                    if d.near_zero() {
                        d = hit.normal;
                    }
                    (albedo, d)
                }
                Material::Metal(albedo, fuzz) => {
                    let unit = dir.unit();
                    let reflected = unit - hit.normal * (2.0 * unit.dot(hit.normal));
                    let d = reflected + rng.in_unit_sphere() * fuzz;
                    if d.dot(hit.normal) <= 0.0 {
                        return V(0.0, 0.0, 0.0);
                    }
                    (albedo, d)
                }
                Material::Glass(ior) => {
                    let ratio = if hit.front { 1.0 / ior } else { ior };
                    let unit = dir.unit();
                    let cos = (-unit).dot(hit.normal).min(1.0);
                    let sin = (1.0 - cos * cos).sqrt();
                    let r0 = ((1.0 - ratio) / (1.0 + ratio)).powi(2);
                    let schlick = r0 + (1.0 - r0) * (1.0 - cos).powi(5);
                    let d = if ratio * sin > 1.0 || schlick > rng.next() {
                        unit - hit.normal * (2.0 * unit.dot(hit.normal))
                    } else {
                        let perp = (unit + hit.normal * cos) * ratio;
                        let par = hit.normal * -(1.0 - perp.dot(perp)).abs().sqrt();
                        perp + par
                    };
                    (V(1.0, 1.0, 1.0), d)
                }
            };
            attenuation = attenuation * albedo;
            origin = hit.point;
            dir = next;
        }
        V(0.0, 0.0, 0.0)
    }

    /// Renders pixel `index` (row-major, row 0 at the top) as packed RGBA
    /// (`r | g << 8 | b << 16 | a << 24`, i.e. bytes R, G, B, A in memory).
    pub fn pixel(&self, index: u32, width: u32, height: u32, spp: u32, depth: u32, seed: u32) -> u32 {
        let x = index % width;
        let y = height - 1 - index / width;
        let mut rng = Rng::new(index ^ hash(seed.wrapping_add(0x9e37_79b9)));
        let mut color = V(0.0, 0.0, 0.0);
        for _ in 0..spp {
            let s = (x as f32 + rng.next()) / (width - 1).max(1) as f32;
            let t = (y as f32 + rng.next()) / (height - 1).max(1) as f32;
            let (origin, dir) = self.camera.ray(s, t, &mut rng);
            color = color + self.trace(origin, dir, depth, &mut rng);
        }
        let to_byte = |c: f32| ((c / spp.max(1) as f32).sqrt().clamp(0.0, 0.999) * 256.0) as u32;
        to_byte(color.0) | (to_byte(color.1) << 8) | (to_byte(color.2) << 16) | (255 << 24)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pixels_do_not_depend_on_how_work_is_split() {
        let (w, h) = (16, 9);
        let scene = Scene::new(42, w as f32 / h as f32);
        let a: Vec<u32> = (0..w * h).map(|i| scene.pixel(i, w, h, 4, 4, 42)).collect();
        // A second scene instance (as on another worker) renders the same pixels.
        let other = Scene::new(42, w as f32 / h as f32);
        let b: Vec<u32> = (0..w * h).rev().map(|i| other.pixel(i, w, h, 4, 4, 42)).collect();
        assert!(a.iter().eq(b.iter().rev()));
        assert!(a.iter().all(|p| p >> 24 == 255));
        // The top row is mostly sky, which is bright.
        assert!((a[0] & 0xff) > 100, "top-left pixel should be sky: {:08x}", a[0]);
    }
}
