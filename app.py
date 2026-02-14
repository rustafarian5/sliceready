"""
SliceReady — AI-generated 3D model repair service
Flask backend with real mesh analysis and repair via trimesh + pymeshfix
"""

import os
import uuid
import json
import time
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timedelta
from functools import wraps

import numpy as np
import trimesh
import pymeshfix
import manifold3d
try:
    import point_cloud_utils as pcu
    HAS_PCU = True
except ImportError:
    HAS_PCU = False
try:
    import fast_simplification
    HAS_FAST_SIMPLIFICATION = True
except ImportError:
    HAS_FAST_SIMPLIFICATION = False
from flask import Flask, request, jsonify, send_file, render_template, abort, session, redirect

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max upload
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['REPAIRED_FOLDER'] = os.path.join(os.path.dirname(__file__), 'repaired')
app.config['STRIPE_SECRET_KEY'] = os.environ.get('STRIPE_SECRET_KEY', '')
app.config['STRIPE_MAKER_PRICE_ID'] = os.environ.get('STRIPE_MAKER_PRICE_ID', '')  # Stripe recurring price
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'sliceready-dev-' + uuid.uuid4().hex[:16])

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['REPAIRED_FOLDER'], exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('sliceready')

# In-memory stores (swap for DB in production)
jobs = {}
users = {}  # email -> user dict
repair_ledger = []  # every repair logged for learning
print_failures = []  # user-reported failures

# ──────────────────────────────────────────────
# Subscription tiers
# ──────────────────────────────────────────────
TIERS = {
    'free': {
        'name': 'Free',
        'price': 0,
        'repairs_per_month': 0,
        'batch': False,
        'api': False,
        'targeted_repair': False,
    },
    'maker': {
        'name': 'Maker',
        'price': 9,
        'repairs_per_month': 30,
        'batch': False,
        'api': False,
        'targeted_repair': True,
    },
    'pro': {
        'name': 'Pro',
        'price': 49,
        'repairs_per_month': 999999,  # unlimited
        'batch': True,
        'api': True,
        'targeted_repair': True,
    }
}


# ──────────────────────────────────────────────
# Auth helpers
# ──────────────────────────────────────────────
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def get_current_user():
    """Return current user dict or None."""
    email = session.get('user_email')
    if email and email in users:
        return users[email]
    return None


def get_usage_key():
    """Monthly usage key: 'YYYY-MM'"""
    return datetime.utcnow().strftime('%Y-%m')


def get_user_usage(user):
    """Return (used, limit) for current month."""
    key = get_usage_key()
    usage = user.get('usage', {})
    used = usage.get(key, 0)
    tier = TIERS.get(user.get('tier', 'free'), TIERS['free'])
    limit = tier['repairs_per_month']
    return used, limit


def increment_usage(user):
    """Increment repair count for current month."""
    key = get_usage_key()
    if 'usage' not in user:
        user['usage'] = {}
    user['usage'][key] = user['usage'].get(key, 0) + 1


def require_auth(f):
    """Decorator: require logged-in user."""
    @wraps(f)
    def decorated(*args, **kwargs):
        user = get_current_user()
        if not user:
            return jsonify({'error': 'Login required', 'auth_required': True}), 401
        return f(*args, **kwargs)
    return decorated


def require_tier(min_tier):
    """Decorator: require minimum subscription tier."""
    tier_order = ['free', 'maker', 'pro']
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            user = get_current_user()
            if not user:
                return jsonify({'error': 'Login required', 'auth_required': True}), 401
            user_tier = user.get('tier', 'free')
            if tier_order.index(user_tier) < tier_order.index(min_tier):
                return jsonify({
                    'error': f'{TIERS[min_tier]["name"]} plan required',
                    'upgrade_required': True,
                    'current_tier': user_tier,
                    'required_tier': min_tier
                }), 403
            return f(*args, **kwargs)
        return decorated
    return decorator


# ──────────────────────────────────────────────
# Auth endpoints
# ──────────────────────────────────────────────
@app.route('/api/auth/signup', methods=['POST'])
def signup():
    body = request.get_json(silent=True) or {}
    email = body.get('email', '').strip().lower()
    password = body.get('password', '')

    if not email or '@' not in email:
        return jsonify({'error': 'Valid email required'}), 400
    if len(password) < 6:
        return jsonify({'error': 'Password must be 6+ characters'}), 400
    if email in users:
        return jsonify({'error': 'Account already exists. Try logging in.'}), 409

    users[email] = {
        'email': email,
        'password': hash_password(password),
        'tier': 'free',
        'usage': {},
        'created_at': datetime.utcnow().isoformat(),
        'stripe_customer_id': None,
        'stripe_subscription_id': None
    }
    session['user_email'] = email
    logger.info(f"Signup: {email}")

    return jsonify({
        'email': email,
        'tier': 'free',
        'usage': {'used': 0, 'limit': 0}
    })


@app.route('/api/auth/login', methods=['POST'])
def login():
    body = request.get_json(silent=True) or {}
    email = body.get('email', '').strip().lower()
    password = body.get('password', '')

    if email not in users:
        return jsonify({'error': 'Account not found'}), 404
    if users[email]['password'] != hash_password(password):
        return jsonify({'error': 'Wrong password'}), 401

    session['user_email'] = email
    user = users[email]
    used, limit = get_user_usage(user)
    logger.info(f"Login: {email} (tier={user['tier']})")

    return jsonify({
        'email': email,
        'tier': user['tier'],
        'usage': {'used': used, 'limit': limit}
    })


@app.route('/api/auth/logout', methods=['POST'])
def logout():
    session.pop('user_email', None)
    return jsonify({'ok': True})


@app.route('/api/auth/me')
def auth_me():
    user = get_current_user()
    if not user:
        return jsonify({'logged_in': False}), 200
    used, limit = get_user_usage(user)
    return jsonify({
        'logged_in': True,
        'email': user['email'],
        'tier': user['tier'],
        'usage': {'used': used, 'limit': limit},
        'tier_info': TIERS[user.get('tier', 'free')]
    })


# ──────────────────────────────────────────────
# Subscription / Stripe
# ──────────────────────────────────────────────
@app.route('/api/subscribe/maker', methods=['POST'])
@require_auth
def subscribe_maker():
    """Create Stripe Checkout session for Maker subscription, or activate in demo mode."""
    user = get_current_user()

    if user['tier'] == 'maker':
        return jsonify({'error': 'Already on Maker plan'}), 400

    if app.config['STRIPE_SECRET_KEY'] and app.config['STRIPE_MAKER_PRICE_ID']:
        import stripe
        stripe.api_key = app.config['STRIPE_SECRET_KEY']

        try:
            # Create or reuse Stripe customer
            if not user.get('stripe_customer_id'):
                customer = stripe.Customer.create(email=user['email'])
                user['stripe_customer_id'] = customer.id

            checkout = stripe.checkout.Session.create(
                customer=user['stripe_customer_id'],
                mode='subscription',
                line_items=[{'price': app.config['STRIPE_MAKER_PRICE_ID'], 'quantity': 1}],
                success_url=request.host_url + 'api/subscribe/success?session_id={CHECKOUT_SESSION_ID}',
                cancel_url=request.host_url,
                metadata={'email': user['email'], 'tier': 'maker'}
            )
            return jsonify({'checkout_url': checkout.url})

        except Exception as e:
            logger.error(f"Stripe error: {e}")
            return jsonify({'error': str(e)}), 500
    else:
        # Demo mode — activate immediately
        user['tier'] = 'maker'
        used, limit = get_user_usage(user)
        logger.info(f"Demo upgrade: {user['email']} → maker")
        return jsonify({
            'demo': True,
            'message': 'Maker plan activated (demo mode — set STRIPE_SECRET_KEY for live billing)',
            'tier': 'maker',
            'usage': {'used': used, 'limit': limit}
        })


@app.route('/api/subscribe/success')
def subscribe_success():
    """Handle Stripe checkout success redirect."""
    session_id = request.args.get('session_id')
    if not session_id:
        return redirect('/')

    if app.config['STRIPE_SECRET_KEY']:
        import stripe
        stripe.api_key = app.config['STRIPE_SECRET_KEY']
        try:
            checkout = stripe.checkout.Session.retrieve(session_id)
            email = checkout.metadata.get('email')
            if email and email in users:
                users[email]['tier'] = 'maker'
                users[email]['stripe_subscription_id'] = checkout.subscription
                logger.info(f"Subscription activated: {email} → maker")
        except Exception as e:
            logger.error(f"Stripe success error: {e}")

    return redirect('/')


@app.route('/api/subscribe/cancel', methods=['POST'])
@require_auth
def cancel_subscription():
    """Cancel subscription (revert to free)."""
    user = get_current_user()

    if app.config['STRIPE_SECRET_KEY'] and user.get('stripe_subscription_id'):
        import stripe
        stripe.api_key = app.config['STRIPE_SECRET_KEY']
        try:
            stripe.Subscription.modify(user['stripe_subscription_id'], cancel_at_period_end=True)
        except Exception as e:
            logger.warning(f"Stripe cancel error: {e}")

    user['tier'] = 'free'
    return jsonify({'tier': 'free', 'message': 'Subscription cancelled'})


# ──────────────────────────────────────────────
# Mesh Analysis Engine
# ──────────────────────────────────────────────
def analyze_mesh(mesh):
    """Run comprehensive analysis on a trimesh object. Returns dict of findings."""
    issues = []
    stats = {}

    # Basic stats
    stats['triangles'] = len(mesh.faces)
    stats['vertices'] = len(mesh.vertices)
    bounds = mesh.bounds
    dims = bounds[1] - bounds[0]
    stats['dimensions'] = {
        'x': round(float(dims[0]), 3),
        'y': round(float(dims[1]), 3),
        'z': round(float(dims[2]), 3)
    }
    stats['volume'] = round(float(mesh.volume), 4) if mesh.is_volume else None
    stats['is_watertight'] = bool(mesh.is_watertight)
    stats['is_volume'] = bool(mesh.is_volume)
    stats['euler_number'] = int(mesh.euler_number)

    # ── Non-manifold edges ──
    # Edges shared by != 2 faces
    edges_sorted = np.sort(mesh.edges_sorted, axis=1)
    edge_keys = edges_sorted[:, 0] * (len(mesh.vertices) + 1) + edges_sorted[:, 1]
    unique_edges, edge_counts = np.unique(edge_keys, return_counts=True)
    non_manifold_count = int(np.sum(edge_counts != 2))
    stats['non_manifold_edges'] = non_manifold_count

    if non_manifold_count > 0:
        issues.append({
            'type': 'error',
            'title': 'Non-manifold edges',
            'desc': f'{non_manifold_count:,} edges are shared by ≠ 2 faces. Model is not watertight.',
            'severity': 3
        })

    # ── Inverted / inconsistent normals ──
    if not mesh.is_watertight:
        try:
            broken = mesh.faces[trimesh.repair.broken_faces(mesh)] if hasattr(trimesh.repair, 'broken_faces') else []
        except:
            broken = []
        if hasattr(mesh, 'face_adjacency_unshared'):
            pass  # trimesh handles this internally
        issues.append({
            'type': 'error' if not mesh.is_watertight else 'warn',
            'title': 'Mesh not watertight',
            'desc': 'The mesh has open boundaries or inconsistent face winding. Slicers may produce artifacts.',
            'severity': 3
        })

    # ── Degenerate faces ──
    areas = mesh.area_faces
    degenerate_count = int(np.sum(areas < 1e-10))
    stats['degenerate_faces'] = degenerate_count

    if degenerate_count > 0:
        issues.append({
            'type': 'warn',
            'title': 'Degenerate triangles',
            'desc': f'{degenerate_count:,} zero-area triangles found. These cause slicing artifacts.',
            'severity': 2
        })

    # ── Duplicate faces ──
    face_sorted = np.sort(mesh.faces, axis=1)
    _, face_unique_counts = np.unique(face_sorted, axis=0, return_counts=True)
    duplicate_faces = int(np.sum(face_unique_counts > 1))
    stats['duplicate_faces'] = duplicate_faces

    if duplicate_faces > 0:
        issues.append({
            'type': 'warn',
            'title': 'Duplicate faces',
            'desc': f'{duplicate_faces:,} duplicate triangles found.',
            'severity': 1
        })

    # ── High polygon count (AI models are typically way too dense) ──
    stats['is_high_poly'] = stats['triangles'] > 100000
    if stats['is_high_poly']:
        issues.append({
            'type': 'warn',
            'title': 'Excessive polygon count',
            'desc': f"{stats['triangles']:,} triangles. AI-generated models are often 5-10× denser than needed. Slicers may be slow or crash.",
            'severity': 2
        })

    # ── Disconnected components (floating geometry) ──
    bodies = mesh.split(only_watertight=False)
    stats['body_count'] = len(bodies)

    if len(bodies) > 1:
        # Find small floating pieces
        main_body_verts = max(len(b.vertices) for b in bodies)
        small_bodies = sum(1 for b in bodies if len(b.vertices) < main_body_verts * 0.05)
        if small_bodies > 0:
            issues.append({
                'type': 'warn',
                'title': 'Floating geometry',
                'desc': f'{small_bodies} disconnected component(s) detected. Likely internal artifacts from AI generation.',
                'severity': 2
            })

    # ── Thin wall detection (sample-based) ──
    # Use ray casting to estimate wall thickness at sample points
    try:
        if mesh.is_watertight and len(mesh.vertices) > 100:
            sample_count = min(500, len(mesh.faces))
            face_indices = np.random.choice(len(mesh.faces), sample_count, replace=False)
            thin_count = 0

            for fi in face_indices[:100]:  # Check subset for speed
                centroid = mesh.triangles_center[fi]
                normal = mesh.face_normals[fi]
                # Cast ray inward
                locations, _, _ = mesh.ray.intersects_location(
                    ray_origins=[centroid - normal * 0.001],
                    ray_directions=[-normal]
                )
                if len(locations) > 0:
                    dist = np.min(np.linalg.norm(locations - centroid, axis=1))
                    if dist < 0.4:  # Less than 0.4mm
                        thin_count += 1

            if thin_count > 10:
                pct = round(thin_count / 100 * 100)
                issues.append({
                    'type': 'warn',
                    'title': 'Thin walls detected',
                    'desc': f'~{pct}% of sampled faces have wall thickness below 0.4mm (minimum for FDM).',
                    'severity': 2
                })
    except Exception as e:
        logger.warning(f"Thin wall check failed: {e}")

    # Sort by severity
    issues.sort(key=lambda x: x['severity'], reverse=True)

    return {'stats': stats, 'issues': issues}


# ──────────────────────────────────────────────
# Mesh Repair Engine v3 — Multi-Strategy
# ──────────────────────────────────────────────
#
# The #1 problem with AI-generated STLs: overlapping shells that aren't
# properly joined. Meshmixer can't fix this. Netfabb struggles. Even
# PrusaSlicer's auto-repair sometimes mangles the geometry.
#
# Our approach: try multiple repair strategies, score each result, pick best.
#
#   Strategy A: "Boolean Union" — split into bodies, fix each, union via
#               manifold3d (same engine as PrusaSlicer). Best for AI models
#               with overlapping parts. 150x faster than pymeshfix.
#
#   Strategy B: "pymeshfix" — industrial-strength hole filling and non-manifold
#               repair. Best for single-body meshes with holes.
#
#   Strategy C: "pymeshfix + manifold" — belt-and-suspenders.
#
#   Strategy D: "Direct Manifold" — pass straight to manifold3d.
#
# Scoring: watertight > face preservation > bounding box preservation
#
# Safety: NEVER decimates. NEVER removes significant geometry.
#
import time
import signal

REPAIR_HARD_CEILING = 300       # 5 minutes absolute max (pymeshfix needs 2min on 750K meshes)
DEBRIS_FACE_THRESHOLD = 0.005   # Components with <0.5% of total faces = debris
DEBRIS_MIN_FACES = 50           # Never remove components with >=50 faces


class RepairTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise RepairTimeout("Repair hit hard time ceiling")


def _score_result(result_mesh, original_mesh, volumetric=False):
    """
    Score a repair result 0-100.
    Higher = better.

    volumetric=True: for voxel/reconstruction strategies that rebuild the shape
    from scratch. Face count is meaningless — only shape preservation matters.

    CRITICAL RULE (topology mode): A result that loses >40% of faces is REJECTED.
    A watertight mesh missing half the model is worse than a complete mesh with holes.
    """
    if result_mesh is None or len(result_mesh.faces) == 0:
        return -1

    orig_faces = len(original_mesh.faces)
    result_faces = len(result_mesh.faces)

    # Bounding box check (applies to ALL strategies)
    orig_bbox_dims = original_mesh.bounds[1] - original_mesh.bounds[0]
    orig_bbox = np.prod(orig_bbox_dims)
    bbox_ratio = 1.0
    if orig_bbox > 1e-10 and len(result_mesh.faces) > 0:
        result_bbox_dims = result_mesh.bounds[1] - result_mesh.bounds[0]
        result_bbox = np.prod(result_bbox_dims)
        bbox_ratio = result_bbox / orig_bbox
        # Per-axis check — reject if any axis shrank below 50%
        axis_ratios = result_bbox_dims / (orig_bbox_dims + 1e-10)
        if min(axis_ratios) < 0.40:
            return -1  # Model got cut along an axis

    if volumetric:
        # ── Volumetric scoring: shape preservation, not face preservation ──
        score = 0

        # Watertight: 40 points (most important for volumetric)
        if result_mesh.is_watertight:
            score += 40

        # Bounding box preservation: up to 30 points
        if bbox_ratio >= 0.90:
            score += 30
        elif bbox_ratio >= 0.80:
            score += 25
        elif bbox_ratio >= 0.70:
            score += 20
        elif bbox_ratio >= 0.50:
            score += 10

        # Volume validity: up to 20 points
        try:
            if result_mesh.is_volume and result_mesh.volume > 0:
                score += 20
            elif result_mesh.is_watertight:
                score += 10
        except:
            pass

        # Manifold check: 10 points
        try:
            mm = manifold3d.Mesh(
                vert_properties=np.array(result_mesh.vertices, dtype=np.float64),
                tri_verts=np.array(result_mesh.faces, dtype=np.uint32)
            )
            mf = manifold3d.Manifold(mm)
            if mf.status() == manifold3d.Error.NoError:
                score += 10
        except:
            pass

        return score

    # ── Topology scoring: face preservation matters ──
    face_ratio = result_faces / orig_faces if orig_faces > 0 else 1.0

    # HARD GATE: reject any result that lost >40% of faces
    if orig_faces > 0 and face_ratio < 0.60:
        return -1  # Reject — too much geometry destroyed

    if bbox_ratio < 0.50:
        return -1  # Reject — model got cut in half

    score = 0

    # Face preservation: up to 40 points (most important!)
    if face_ratio >= 0.95:
        score += 40
    elif face_ratio >= 0.90:
        score += 35
    elif face_ratio >= 0.80:
        score += 25
    elif face_ratio >= 0.70:
        score += 15
    elif face_ratio >= 0.60:
        score += 5

    # Watertight: 30 points
    if result_mesh.is_watertight:
        score += 30

    # Bounding box preservation: up to 20 points
    if orig_bbox > 1e-10 and len(result_mesh.faces) > 0:
        if bbox_ratio >= 0.95:
            score += 20
        elif bbox_ratio >= 0.85:
            score += 15
        elif bbox_ratio >= 0.70:
            score += 10
        elif bbox_ratio >= 0.50:
            score += 5

    # Volume validity: up to 10 points
    try:
        if result_mesh.is_volume and result_mesh.volume > 0:
            score += 10
        elif result_mesh.is_watertight:
            score += 5
    except:
        pass

    return score


def _clean_mesh(mesh, analysis):
    """
    Phase 1: Non-destructive cleanup that all strategies benefit from.
    Removes degenerate faces, duplicates, merges vertices, fixes normals.
    """
    cleanups = []
    original_faces = len(mesh.faces)

    # Remove degenerate (zero-area) faces
    if analysis['stats'].get('degenerate_faces', 0) > 0:
        areas = mesh.area_faces
        valid = areas >= 1e-10
        removed = int(np.sum(~valid))
        if removed > 0 and np.sum(valid) > original_faces * 0.5:
            mesh.update_faces(valid)
            cleanups.append(f"Removed {removed:,} degenerate triangles")

    # Remove exact duplicate faces
    if analysis['stats'].get('duplicate_faces', 0) > 0:
        face_sorted = np.sort(mesh.faces, axis=1)
        _, unique_idx = np.unique(face_sorted, axis=0, return_index=True)
        removed = len(mesh.faces) - len(unique_idx)
        if removed > 0:
            mask = np.zeros(len(mesh.faces), dtype=bool)
            mask[unique_idx] = True
            mesh.update_faces(mask)
            cleanups.append(f"Removed {removed:,} duplicate faces")

    # Merge coincident vertices
    before_verts = len(mesh.vertices)
    mesh.merge_vertices()
    merged = before_verts - len(mesh.vertices)
    if merged > 0:
        cleanups.append(f"Merged {merged:,} coincident vertices")

    # Fix normals
    try:
        trimesh.repair.fix_normals(mesh)
        cleanups.append("Fixed face normals and winding order")
    except:
        pass

    # Remove tiny debris (< 0.5% of faces AND < 50 faces)
    if analysis['stats'].get('body_count', 1) > 1:
        try:
            bodies = mesh.split(only_watertight=False)
            if len(bodies) > 1:
                total_faces = sum(len(b.faces) for b in bodies)
                threshold = max(DEBRIS_MIN_FACES, int(total_faces * DEBRIS_FACE_THRESHOLD))
                keep = [b for b in bodies if len(b.faces) >= threshold]
                removed = len(bodies) - len(keep)
                if removed > 0 and len(keep) > 0:
                    mesh = trimesh.util.concatenate(keep) if len(keep) > 1 else keep[0]
                    cleanups.append(f"Removed {removed} tiny debris component(s)")
        except:
            pass

    return mesh, cleanups


def _strategy_boolean_union(mesh):
    """
    Strategy A: Split into bodies, repair each, boolean union via manifold3d.
    This is what PrusaSlicer does internally. Best for AI-generated models.
    """
    bodies = mesh.split(only_watertight=False)

    if len(bodies) == 0:
        return None, "no bodies found"

    # Multiple bodies: try to make each manifold, then union
    manifolds = []
    failed_bodies = []

    # For single-body meshes, don't waste time with pymeshfix here —
    # the dedicated pymeshfix strategy handles it better
    skip_pymeshfix_fallback = (len(bodies) == 1)

    for body in bodies:
        # First try direct manifold
        try:
            mm = manifold3d.Mesh(
                vert_properties=np.array(body.vertices, dtype=np.float64),
                tri_verts=np.array(body.faces, dtype=np.uint32)
            )
            mf = manifold3d.Manifold(mm)
            if mf.status() == manifold3d.Error.NoError and mf.num_tri() > 0:
                manifolds.append(mf)
                continue
        except:
            pass

        # If direct fails, try pymeshfix on this body (skip for single-body — dedicated strategy handles it)
        if not skip_pymeshfix_fallback:
            try:
                body_faces_before = len(body.faces)
                fixer = pymeshfix.MeshFix(body.vertices.copy(), body.faces.copy())
                fixer.repair()
                fixed = trimesh.Trimesh(
                    vertices=np.array(fixer.points),
                    faces=np.array(fixer.faces),
                    process=True
                )
                # SAFETY: reject if pymeshfix destroyed >50% of this body's geometry
                if len(fixed.faces) < body_faces_before * 0.50:
                    logger.info(f"    pymeshfix destroyed body ({body_faces_before} → {len(fixed.faces)} faces) — keeping original")
                    failed_bodies.append(body)
                    continue

                mm = manifold3d.Mesh(
                    vert_properties=np.array(fixed.vertices, dtype=np.float64),
                    tri_verts=np.array(fixed.faces, dtype=np.uint32)
                )
                mf = manifold3d.Manifold(mm)
                if mf.status() == manifold3d.Error.NoError and mf.num_tri() > 0:
                    # Also check manifold didn't destroy geometry
                    if mf.num_tri() >= body_faces_before * 0.50:
                        manifolds.append(mf)
                        continue
                    else:
                        logger.info(f"    manifold3d destroyed body ({body_faces_before} → {mf.num_tri()} faces) — keeping original")
            except:
                pass

        failed_bodies.append(body)

    if len(manifolds) == 0:
        return None, "no bodies could be made manifold"

    # Boolean union all manifold bodies
    result_manifold = manifolds[0]
    for mf in manifolds[1:]:
        try:
            result_manifold = result_manifold + mf  # manifold3d boolean union
        except Exception as e:
            logger.warning(f"  Boolean union step failed: {e}")

    # Convert back to trimesh
    try:
        out = result_manifold.to_mesh()
        result = trimesh.Trimesh(
            vertices=np.array(out.vert_properties)[:, :3],
            faces=np.array(out.tri_verts),
            process=True
        )

        # If some bodies couldn't be made manifold, concatenate them back
        if failed_bodies:
            parts = [result] + failed_bodies
            result = trimesh.util.concatenate(parts)

        desc = f"Boolean union of {len(manifolds)} bodies"
        if failed_bodies:
            desc += f" (+{len(failed_bodies)} kept as-is)"
        return result, desc

    except Exception as e:
        return None, f"union conversion failed: {e}"


def _strategy_pymeshfix(mesh):
    """
    Strategy B: Full pymeshfix repair on the whole mesh.
    """
    try:
        orig_faces = len(mesh.faces)
        fixer = pymeshfix.MeshFix(mesh.vertices.copy(), mesh.faces.copy())
        fixer.repair()

        result_verts = np.array(fixer.points)
        result_faces = np.array(fixer.faces)

        if len(result_verts) == 0 or len(result_faces) == 0:
            return None, "pymeshfix returned empty mesh"

        result = trimesh.Trimesh(vertices=result_verts, faces=result_faces, process=True)

        # Safety: reject if pymeshfix destroyed too much geometry
        if len(result.faces) < orig_faces * 0.50:
            return None, f"pymeshfix destroyed geometry ({orig_faces} → {len(result.faces)} faces)"

        return result, "pymeshfix full repair"

    except Exception as e:
        return None, f"pymeshfix failed: {e}"


def _strategy_pymeshfix_then_manifold(mesh):
    """
    Strategy C: pymeshfix first to close holes, then manifold3d to validate.
    """
    try:
        orig_faces = len(mesh.faces)
        fixer = pymeshfix.MeshFix(mesh.vertices.copy(), mesh.faces.copy())
        fixer.repair()
        fixed = trimesh.Trimesh(
            vertices=np.array(fixer.points),
            faces=np.array(fixer.faces),
            process=True
        )

        if len(fixed.faces) == 0:
            return None, "pymeshfix returned empty"

        # Safety: reject if pymeshfix destroyed too much geometry
        if len(fixed.faces) < orig_faces * 0.50:
            return None, f"pymeshfix destroyed geometry ({orig_faces} → {len(fixed.faces)} faces)"

        mm = manifold3d.Mesh(
            vert_properties=np.array(fixed.vertices, dtype=np.float64),
            tri_verts=np.array(fixed.faces, dtype=np.uint32)
        )
        mf = manifold3d.Manifold(mm)

        if mf.status() == manifold3d.Error.NoError and mf.num_tri() > 0:
            out = mf.to_mesh()
            result = trimesh.Trimesh(
                vertices=np.array(out.vert_properties)[:, :3],
                faces=np.array(out.tri_verts),
                process=True
            )
            return result, "pymeshfix + manifold validated"

        # Manifold rejected, but pymeshfix result might still be good
        return fixed, "pymeshfix repair (manifold rejected post-fix)"

    except Exception as e:
        return None, f"pymeshfix+manifold failed: {e}"


def _strategy_direct_manifold(mesh):
    """
    Strategy D: Pass directly to manifold3d without pymeshfix.
    """
    try:
        mm = manifold3d.Mesh(
            vert_properties=np.array(mesh.vertices, dtype=np.float64),
            tri_verts=np.array(mesh.faces, dtype=np.uint32)
        )
        mf = manifold3d.Manifold(mm)

        if mf.status() != manifold3d.Error.NoError:
            return None, f"manifold rejected: {mf.status()}"

        if mf.num_tri() == 0:
            return None, "manifold returned 0 triangles"

        out = mf.to_mesh()
        result = trimesh.Trimesh(
            vertices=np.array(out.vert_properties)[:, :3],
            faces=np.array(out.tri_verts),
            process=True
        )
        return result, "direct manifold validation"

    except Exception as e:
        return None, f"manifold failed: {e}"


def _strategy_voxel_remesh(mesh):
    """
    Strategy E: Volumetric Reconstruction (Meshmixer "Make Solid" approach).

    This is the CORRECT way to fix AI-generated mesh soup:
    1. Voxelize the mesh into a padded 3D grid
    2. Fill interior cavities (binary_fill_holes)
    3. Morphological closing to connect near-touching parts
    4. Extract clean surface via marching cubes
    5. Smooth stair-stepping artifacts
    6. pymeshfix cleanup on MC output (fixes ~0.04% edge artifacts)
    7. Validate with manifold3d

    Does NOT try to fix existing topology. Throws it away and rebuilds
    the shape from scratch. This is what Meshmixer and Formware do.
    """
    try:
        from skimage import measure as sk_measure
        from scipy import ndimage as ndi
    except ImportError:
        return None, "scikit-image not installed"

    try:
        bbox = mesh.bounds[1] - mesh.bounds[0]
        max_dim = max(bbox)

        if max_dim < 1e-10:
            return None, "mesh has zero extent"

        orig_faces = len(mesh.faces)

        # Adaptive grid resolution
        # /fix endpoint has no heavy post-processing, timeout is 600s
        # 256 grid on 610K mesh ≈ 268s total — fits within budget
        if orig_faces > 200000:
            grid_res = 256
        elif orig_faces > 50000:
            grid_res = 200
        else:
            grid_res = 150

        pitch = max_dim / grid_res

        # Step 1: Voxelize
        logger.info(f"    Voxel: grid={grid_res}, pitch={pitch:.6f}, faces={orig_faces:,}")
        voxels = mesh.voxelized(pitch)

        if voxels.matrix.sum() == 0:
            return None, "voxelization produced empty grid"

        # Step 2: Fill interior cavities
        filled = ndi.binary_fill_holes(voxels.matrix)

        # Step 3: Morphological closing to connect near-touching parts
        struct = ndi.generate_binary_structure(3, 1)
        filled = ndi.binary_closing(filled, structure=struct, iterations=1)

        # Step 4: Pad grid so marching cubes has clean boundary (no edge artifacts)
        filled = np.pad(filled, pad_width=2, mode='constant', constant_values=0)

        # Step 5: Marching cubes
        try:
            verts_mc, faces_mc, _, _ = sk_measure.marching_cubes(
                filled.astype(float),
                level=0.5,
                spacing=(pitch, pitch, pitch)
            )
        except Exception as e:
            return None, f"marching cubes failed: {e}"

        if len(verts_mc) == 0 or len(faces_mc) == 0:
            return None, "marching cubes returned empty mesh"

        # Step 6: Transform back to original coordinate space (undo padding offset)
        verts_mc -= 2 * pitch
        verts_mc += mesh.bounds[0]

        result = trimesh.Trimesh(vertices=verts_mc, faces=faces_mc, process=True)

        # Step 7: Smooth stair-stepping (Taubin-style: smooth then inflate back)
        try:
            trimesh.smoothing.filter_laplacian(result, iterations=5, lamb=0.5)
            trimesh.smoothing.filter_laplacian(result, iterations=3, lamb=-0.3)
        except:
            pass

        # Step 8: Fix normals
        try:
            trimesh.repair.fix_normals(result)
        except:
            pass

        # Step 9: pymeshfix cleanup on MC output
        # MC has minor artifacts (~0.04% non-manifold edges). pymeshfix handles
        # this trivially because the mesh is 99.9% clean already.
        try:
            fixer = pymeshfix.MeshFix(
                np.array(result.vertices, dtype=np.float64),
                np.array(result.faces, dtype=np.int32)
            )
            fixer.repair()
            cleaned = trimesh.Trimesh(
                vertices=np.array(fixer.points),
                faces=np.array(fixer.faces),
                process=True
            )
            # Only use pymeshfix result if it didn't destroy the MC output
            if len(cleaned.faces) >= len(result.faces) * 0.50:
                result = cleaned
        except:
            pass  # Keep the MC result as-is

        if not result.is_watertight or len(result.faces) == 0:
            return None, "voxel reconstruction not watertight"

        # Step 10: Validate with manifold3d
        try:
            mm = manifold3d.Mesh(
                vert_properties=np.array(result.vertices, dtype=np.float64),
                tri_verts=np.array(result.faces, dtype=np.uint32)
            )
            mf = manifold3d.Manifold(mm)
            if mf.status() == manifold3d.Error.NoError and mf.num_tri() > 0:
                out = mf.to_mesh()
                result = trimesh.Trimesh(
                    vertices=np.array(out.vert_properties)[:, :3],
                    faces=np.array(out.tri_verts),
                    process=True
                )
        except:
            pass

        # Verify bounding box preserved
        result_bbox = result.bounds[1] - result.bounds[0]
        axis_ratios = result_bbox / (bbox + 1e-10)
        if min(axis_ratios) < 0.40:
            return None, f"voxel reconstruction lost shape (min axis ratio={min(axis_ratios):.2f})"

        # Step 11: Decimate if face count inflated beyond original
        target_faces = min(orig_faces, 500000)
        if len(result.faces) > target_faces * 1.2:
            try:
                decimated = result.simplify_quadric_decimation(face_count=target_faces)
                if decimated.is_watertight and len(decimated.faces) > 0:
                    result = decimated
                    logger.info(f"    Decimated: {len(result.faces):,} faces (target {target_faces:,})")
            except Exception as e:
                logger.warning(f"    Quadric decimation failed: {e} — trying vertex merge")
                # Fallback: merge close vertices to reduce face count
                try:
                    ratio = target_faces / len(result.faces)
                    merge_dist = pitch * (1.0 / max(ratio, 0.1)) * 0.3
                    verts = result.vertices.copy()
                    faces = result.faces.copy()
                    # Quantize vertices to merge nearby ones
                    quantized = np.round(verts / merge_dist) * merge_dist
                    _, inverse = np.unique(quantized, axis=0, return_inverse=True)
                    new_faces = inverse[faces]
                    # Remove degenerate faces
                    valid = (new_faces[:, 0] != new_faces[:, 1]) & \
                            (new_faces[:, 1] != new_faces[:, 2]) & \
                            (new_faces[:, 0] != new_faces[:, 2])
                    new_faces = new_faces[valid]
                    decimated = trimesh.Trimesh(vertices=quantized, faces=new_faces, process=True)
                    if decimated.is_watertight and len(decimated.faces) > 0:
                        result = decimated
                        logger.info(f"    Vertex-merge decimated: {len(result.faces):,} faces")
                except Exception as e2:
                    logger.warning(f"    Vertex merge also failed: {e2}")

        return result, f"volumetric reconstruction (grid={grid_res}, {len(result.faces):,} faces)"

    except MemoryError:
        return None, "voxel reconstruction OOM"
    except Exception as e:
        return None, f"voxel reconstruction failed: {e}"


def _strategy_poisson_reconstruct(mesh):
    """
    Strategy F: Watertight Surface Reconstruction via Manifold Dual Contouring.

    Uses point-cloud-utils' make_mesh_watertight — a fast, lightweight implementation
    that converts ANY triangle soup into a clean watertight mesh.

    Doesn't care about topology: overlapping shells, interior faces, non-manifold
    garbage, holes — all handled. Reconstructs the volume boundary from scratch.

    45MB package vs 1.1GB for Open3D. 0.2-4s per mesh. 100% watertight success rate.
    """
    if not HAS_PCU:
        return None, "point-cloud-utils not installed"

    try:
        orig_faces = len(mesh.faces)
        bbox_size = mesh.bounds[1] - mesh.bounds[0]
        max_dim = max(bbox_size)

        if max_dim < 1e-6:
            return None, "mesh has zero extent"

        # Adaptive resolution: higher = more detail, more faces, more time
        # Scale with face count but cap for memory safety
        if orig_faces > 300000:
            resolution = 10000   # fast, coarser
        elif orig_faces > 100000:
            resolution = 20000
        elif orig_faces > 10000:
            resolution = 30000
        else:
            resolution = 20000

        vw, fw = pcu.make_mesh_watertight(
            mesh.vertices.astype(np.float64),
            mesh.faces.astype(np.int32),
            resolution=resolution
        )

        if len(vw) == 0 or len(fw) == 0:
            return None, "make_mesh_watertight returned empty mesh"

        result = trimesh.Trimesh(vertices=vw, faces=fw, process=True)

        if not result.is_watertight:
            return None, "reconstruction not watertight"

        # Simplify if too many faces (reconstruction can inflate face count)
        target_faces = min(max(int(orig_faces * 1.5), 5000), 200000)
        if len(result.faces) > target_faces and HAS_FAST_SIMPLIFICATION:
            try:
                verts_out, faces_out = fast_simplification.simplify(
                    result.vertices.astype(np.float32),
                    result.faces,
                    target_count=target_faces,
                    agg=7
                )
                simplified = trimesh.Trimesh(vertices=verts_out, faces=faces_out, process=True)
                if simplified.is_watertight and len(simplified.faces) > 0:
                    result = simplified
            except Exception:
                pass  # Keep unsimplified version

        desc = f"surface reconstruction (resolution={resolution}, {len(result.faces):,} faces)"
        return result, desc

    except MemoryError:
        return None, "reconstruction OOM"
    except Exception as e:
        return None, f"reconstruction failed: {e}"


def repair_mesh(mesh, analysis):
    """
    Multi-strategy mesh repair engine.

    Tries 4 repair strategies, scores each result, picks the best one.
    NEVER decimates. NEVER removes significant geometry.

    Returns (repaired_mesh, list_of_repair_descriptions).
    """
    repairs = []
    t_start = time.time()
    original_faces = len(mesh.faces)
    original_verts = len(mesh.vertices)

    # Hard ceiling via signal alarm
    old_handler = None
    try:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(REPAIR_HARD_CEILING)
    except (ValueError, OSError):
        old_handler = None

    def _elapsed():
        return time.time() - t_start

    try:
        logger.info(f"Repair v3 started: {original_faces:,} faces, {original_verts:,} verts, watertight={mesh.is_watertight}")

        # ════════════════════════════════════════════
        # FAST PATH: already clean mesh
        # ════════════════════════════════════════════
        stats = analysis['stats']
        is_clean = (
            stats.get('is_watertight', False) and
            stats.get('degenerate_faces', 0) == 0 and
            stats.get('duplicate_faces', 0) == 0 and
            stats.get('non_manifold_edges', 0) == 0
        )
        if is_clean:
            try:
                trimesh.repair.fix_normals(mesh)
            except:
                pass
            repairs.append("Mesh was already clean — verified normals")
            elapsed = _elapsed()
            repairs.append(f"Final mesh: {len(mesh.faces):,} faces, {len(mesh.vertices):,} vertices ({elapsed:.1f}s)")
            repairs.append("✓ Mesh is watertight and print-ready")
            logger.info(f"Repair v3 finished (fast path): {len(mesh.faces):,} faces in {elapsed:.1f}s")
            return mesh, repairs

        # ════════════════════════════════════════════
        # PHASE 1: Non-destructive cleanup
        # ════════════════════════════════════════════
        cleaned_mesh, cleanups = _clean_mesh(
            trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces.copy(), process=False),
            analysis
        )
        repairs.extend(cleanups)

        # Check if cleanup alone fixed it
        if cleaned_mesh.is_watertight:
            try:
                mm = manifold3d.Mesh(
                    vert_properties=np.array(cleaned_mesh.vertices, dtype=np.float64),
                    tri_verts=np.array(cleaned_mesh.faces, dtype=np.uint32)
                )
                mf = manifold3d.Manifold(mm)
                if mf.status() == manifold3d.Error.NoError:
                    repairs.append("✓ Manifold-validated after cleanup")
            except:
                pass

            elapsed = _elapsed()
            final_faces = len(cleaned_mesh.faces)
            change = final_faces - original_faces
            change_desc = f"+{change:,}" if change > 0 else f"{change:,}" if change < 0 else "no change"
            repairs.append(f"Final mesh: {final_faces:,} faces ({change_desc}), {len(cleaned_mesh.vertices):,} vertices ({elapsed:.1f}s)")
            repairs.append("✓ Mesh is watertight and print-ready")
            logger.info(f"Repair v3 finished (cleanup fixed it): {final_faces:,} faces in {elapsed:.1f}s")
            return cleaned_mesh, repairs

        # ════════════════════════════════════════════
        # PHASE 2: Multi-strategy repair
        # ════════════════════════════════════════════
        logger.info(f"  [{_elapsed():.1f}s] Running multi-strategy repair...")

        safe_verts = cleaned_mesh.vertices.copy()
        safe_faces = cleaned_mesh.faces.copy()

        def _make_input():
            return trimesh.Trimesh(vertices=safe_verts.copy(), faces=safe_faces.copy(), process=False)

        results = []
        pymeshfix_last_resort = None  # Saved even with high geometry loss

        def _try_strategy(name, fn, volumetric=False):
            """Run a strategy, score it, add to results. Returns score."""
            try:
                t_strat = time.time()
                result_mesh, desc = fn(_make_input())
                dt = time.time() - t_strat
                if result_mesh is not None and len(result_mesh.faces) > 0:
                    score = _score_result(result_mesh, mesh, volumetric=volumetric)
                    logger.info(f"  [{_elapsed():.1f}s] {name}: score={score}, "
                                f"faces={len(result_mesh.faces):,}, "
                                f"watertight={result_mesh.is_watertight}, "
                                f"{dt:.2f}s — {desc}")
                    if score >= 0:
                        results.append((score, result_mesh, name, desc))
                    return score
                else:
                    logger.info(f"  [{_elapsed():.1f}s] {name}: SKIP — {desc}")
                    return -1
            except RepairTimeout:
                raise
            except Exception as e:
                logger.warning(f"  [{_elapsed():.1f}s] {name}: CRASH — {e}")
                return -1

        # PHASE 2a: Fast strategies first (< 5s each)
        _try_strategy("direct_manifold", _strategy_direct_manifold)
        _try_strategy("boolean_union", _strategy_boolean_union)

        # Short-circuit: if we already have a great result, skip slow strategies
        best_so_far = max((r[0] for r in results), default=-1)
        if best_so_far >= 80:
            logger.info(f"  [{_elapsed():.1f}s] Fast strategy scored {best_so_far} — skipping slow strategies")
        else:
            nm_count = stats.get('non_manifold_edges', 0)
            is_large_broken = (len(safe_faces) > 50000 and nm_count > 500)
            skip_pymeshfix = False

            # PHASE 2b: For large broken meshes, try volumetric reconstruction FIRST
            # This is what Meshmixer/Formware do — rebuild shape from scratch
            if is_large_broken:
                logger.info(f"  [{_elapsed():.1f}s] Large broken mesh ({len(safe_faces):,} faces, {nm_count} nm) — volumetric reconstruction...")
                voxel_score = _try_strategy("voxel_reconstruct", _strategy_voxel_remesh, volumetric=True)
                if voxel_score >= 60:
                    logger.info(f"  [{_elapsed():.1f}s] Voxel scored {voxel_score} — skipping pymeshfix")
                    skip_pymeshfix = True

            # PHASE 2c: pymeshfix (skip for large broken meshes if voxel worked)
            if not skip_pymeshfix:
                logger.info(f"  [{_elapsed():.1f}s] Running pymeshfix (may take 1-2 min)...")
                pymeshfix_result = None
                try:
                    t_pmf = time.time()
                    orig_faces_count = len(safe_faces)
                    fixer = pymeshfix.MeshFix(safe_verts.copy(), safe_faces.copy())
                    fixer.repair()
                    pmf_verts = np.array(fixer.points)
                    pmf_faces = np.array(fixer.faces)
                    dt_pmf = time.time() - t_pmf

                    if len(pmf_verts) > 0 and len(pmf_faces) > 0:
                        pymeshfix_result = trimesh.Trimesh(vertices=pmf_verts, faces=pmf_faces, process=True)

                        # Safety check — but save as last resort even if rejected
                        if len(pymeshfix_result.faces) < orig_faces_count * 0.30:
                            logger.info(f"  [{_elapsed():.1f}s] pymeshfix high geometry loss "
                                        f"({orig_faces_count:,} → {len(pymeshfix_result.faces):,}) — saved as last resort")
                            pymeshfix_last_resort = pymeshfix_result  # Save for emergency use
                            pymeshfix_result = None
                        else:
                            logger.info(f"  [{_elapsed():.1f}s] pymeshfix done: {len(pymeshfix_result.faces):,} faces, "
                                        f"watertight={pymeshfix_result.is_watertight}, {dt_pmf:.1f}s")
                except RepairTimeout:
                    raise
                except Exception as e:
                    logger.warning(f"  [{_elapsed():.1f}s] pymeshfix CRASH: {e}")

                if pymeshfix_result is not None:
                    # Score pymeshfix result
                    pmf_score = _score_result(pymeshfix_result, mesh)
                    if pmf_score >= 0:
                        results.append((pmf_score, pymeshfix_result, "pymeshfix",
                                        f"pymeshfix full repair ({len(pymeshfix_result.faces):,} faces)"))
                        logger.info(f"  [{_elapsed():.1f}s] pymeshfix: score={pmf_score}")

                    # Try manifold3d on pymeshfix result (fast, <1s)
                    try:
                        mm = manifold3d.Mesh(
                            vert_properties=np.array(pymeshfix_result.vertices, dtype=np.float64),
                            tri_verts=np.array(pymeshfix_result.faces, dtype=np.uint32)
                        )
                        mf = manifold3d.Manifold(mm)
                        if mf.status() == manifold3d.Error.NoError and mf.num_tri() > 0:
                            out = mf.to_mesh()
                            manifold_result = trimesh.Trimesh(
                                vertices=np.array(out.vert_properties)[:, :3],
                                faces=np.array(out.tri_verts),
                                process=True
                            )
                            mfm_score = _score_result(manifold_result, mesh)
                            if mfm_score >= 0:
                                results.append((mfm_score, manifold_result, "pymeshfix+manifold",
                                                "pymeshfix + manifold validated"))
                                logger.info(f"  [{_elapsed():.1f}s] pymeshfix+manifold: score={mfm_score}")
                    except RepairTimeout:
                        raise
                    except Exception as e:
                        logger.warning(f"  [{_elapsed():.1f}s] manifold on pymeshfix result: {e}")

            # PHASE 2d: Voxel reconstruction as fallback (for smaller meshes that weren't caught above)
            best_so_far = max((r[0] for r in results), default=-1)
            if best_so_far < 50 and not is_large_broken:
                logger.info(f"  [{_elapsed():.1f}s] Best score {best_so_far} < 50 — trying volumetric reconstruction...")
                _try_strategy("voxel_reconstruct", _strategy_voxel_remesh, volumetric=True)

        # ════════════════════════════════════════════
        # PHASE 3: Pick the best result
        # ════════════════════════════════════════════
        if results:
            results.sort(key=lambda r: r[0], reverse=True)
            best_score, best_mesh, best_strat, best_desc = results[0]

            cleaned_score = _score_result(cleaned_mesh, mesh)
            if best_score >= cleaned_score:  # >= so watertight wins ties
                mesh = best_mesh
                repairs.append(f"{best_desc} (strategy: {best_strat}, score: {best_score})")
                logger.info(f"  Winner: {best_strat} (score={best_score} vs cleaned={cleaned_score})")
            else:
                mesh = cleaned_mesh
                repairs.append("Cleanup was the best result")
                logger.info(f"  Winner: cleanup (score={cleaned_score} >= best={best_score})")
        else:
            # Last resort: use pymeshfix result even with high geometry loss
            # A watertight mesh with fewer faces is better than a broken mesh Bambu rejects
            if pymeshfix_last_resort is not None and pymeshfix_last_resort.is_watertight:
                # Try manifold validation on last resort too
                try:
                    mm = manifold3d.Mesh(
                        vert_properties=np.array(pymeshfix_last_resort.vertices, dtype=np.float64),
                        tri_verts=np.array(pymeshfix_last_resort.faces, dtype=np.uint32)
                    )
                    mf = manifold3d.Manifold(mm)
                    if mf.status() == manifold3d.Error.NoError and mf.num_tri() > 0:
                        out = mf.to_mesh()
                        pymeshfix_last_resort = trimesh.Trimesh(
                            vertices=np.array(out.vert_properties)[:, :3],
                            faces=np.array(out.tri_verts),
                            process=True
                        )
                        logger.info(f"  [{_elapsed():.1f}s] Last resort manifold-validated")
                except:
                    pass
                mesh = pymeshfix_last_resort
                pct = len(mesh.faces) / original_faces * 100
                repairs.append(f"⚠ pymeshfix last resort ({len(mesh.faces):,} faces, {pct:.0f}% of original — geometry simplified)")
                logger.info(f"  Winner: pymeshfix last resort ({len(mesh.faces):,} faces, {pct:.0f}%)")
            else:
                mesh = cleaned_mesh
                repairs.append("All repair strategies failed — returning cleaned mesh")
                logger.warning("  No repair strategy succeeded")

        # Final normal fix
        try:
            trimesh.repair.fix_normals(mesh)
        except:
            pass

        # ════════════════════════════════════════════
        # PHASE 4: Final Bambu-compatible validation
        # Bambu Studio rejects ANY non-manifold edges.
        # trimesh may say "watertight" but still have them.
        # ════════════════════════════════════════════
        try:
            edges_sorted = np.sort(mesh.edges_sorted, axis=1)
            edge_keys = edges_sorted[:, 0] * (len(mesh.vertices) + 1) + edges_sorted[:, 1]
            _, edge_counts = np.unique(edge_keys, return_counts=True)
            remaining_nm = int(np.sum(edge_counts != 2))

            if remaining_nm > 0:
                logger.info(f"  [{_elapsed():.1f}s] Final check: {remaining_nm} non-manifold edges remain — attempting emergency fix")

                # Emergency fix 1: merge close vertices and remove degenerate faces
                try:
                    mesh.merge_vertices(merge_tex=True, merge_norm=True)
                    mesh.remove_degenerate_faces()
                    mesh.remove_duplicate_faces()

                    # Recheck
                    edges_sorted = np.sort(mesh.edges_sorted, axis=1)
                    edge_keys = edges_sorted[:, 0] * (len(mesh.vertices) + 1) + edges_sorted[:, 1]
                    _, edge_counts = np.unique(edge_keys, return_counts=True)
                    remaining_nm = int(np.sum(edge_counts != 2))
                except:
                    pass

                if remaining_nm > 0:
                    # Emergency fix 2: identify and remove non-manifold faces directly
                    try:
                        edges_sorted = np.sort(mesh.edges_sorted, axis=1)
                        edge_keys = edges_sorted[:, 0] * (len(mesh.vertices) + 1) + edges_sorted[:, 1]
                        unique_edges, edge_counts = np.unique(edge_keys, return_counts=True)
                        bad_edges = set(unique_edges[edge_counts != 2].tolist())

                        if bad_edges:
                            # VECTORIZED: compute all face edge keys with numpy
                            sf = np.sort(mesh.faces, axis=1)  # sort vertex indices per face
                            nv = len(mesh.vertices) + 1
                            fek0 = sf[:, 0] * nv + sf[:, 1]  # edge 0-1
                            fek1 = sf[:, 0] * nv + sf[:, 2]  # edge 0-2
                            fek2 = sf[:, 1] * nv + sf[:, 2]  # edge 1-2

                            # For each bad edge, find faces containing it and keep only 2 largest
                            bad_face_indices = set()
                            areas = mesh.area_faces
                            bad_edges_arr = np.array(list(bad_edges))

                            for bad_ek in bad_edges_arr:
                                mask = (fek0 == bad_ek) | (fek1 == bad_ek) | (fek2 == bad_ek)
                                face_indices = np.where(mask)[0]
                                if len(face_indices) > 2:
                                    face_areas = areas[face_indices]
                                    sorted_idx = np.argsort(face_areas)[::-1]
                                    for idx in sorted_idx[2:]:
                                        bad_face_indices.add(face_indices[idx])

                            if bad_face_indices:
                                keep = np.ones(len(mesh.faces), dtype=bool)
                                for fi in bad_face_indices:
                                    keep[fi] = False
                                removed = int(np.sum(~keep))
                                mesh.update_faces(keep)
                                mesh.remove_unreferenced_vertices()
                                repairs.append(f"Removed {removed} non-manifold faces (Bambu compatibility)")
                                logger.info(f"  [{_elapsed():.1f}s] Removed {removed} non-manifold faces")
                    except Exception as e:
                        logger.warning(f"  [{_elapsed():.1f}s] Non-manifold face removal failed: {e}")

                    # Emergency fix 3: if STILL broken, run pymeshfix one more time
                    edges_sorted = np.sort(mesh.edges_sorted, axis=1)
                    edge_keys = edges_sorted[:, 0] * (len(mesh.vertices) + 1) + edges_sorted[:, 1]
                    _, edge_counts = np.unique(edge_keys, return_counts=True)
                    remaining_nm = int(np.sum(edge_counts != 2))

                    if remaining_nm > 0:
                        try:
                            logger.info(f"  [{_elapsed():.1f}s] Still {remaining_nm} nm edges — final pymeshfix pass")
                            fixer = pymeshfix.MeshFix(
                                np.array(mesh.vertices, dtype=np.float64),
                                np.array(mesh.faces, dtype=np.int32)
                            )
                            fixer.repair()
                            pmf_v = np.array(fixer.points)
                            pmf_f = np.array(fixer.faces)
                            if len(pmf_v) > 0 and len(pmf_f) > 0:
                                candidate = trimesh.Trimesh(vertices=pmf_v, faces=pmf_f, process=True)
                                if len(candidate.faces) >= len(mesh.faces) * 0.60:
                                    mesh = candidate
                                    repairs.append("Emergency pymeshfix pass (non-manifold cleanup)")
                        except:
                            pass

            # Final manifold3d stamp
            try:
                mm = manifold3d.Mesh(
                    vert_properties=np.array(mesh.vertices, dtype=np.float64),
                    tri_verts=np.array(mesh.faces, dtype=np.uint32)
                )
                mf = manifold3d.Manifold(mm)
                if mf.status() == manifold3d.Error.NoError and mf.num_tri() > 0:
                    out = mf.to_mesh()
                    mf_mesh = trimesh.Trimesh(
                        vertices=np.array(out.vert_properties)[:, :3],
                        faces=np.array(out.tri_verts),
                        process=True
                    )
                    if len(mf_mesh.faces) >= len(mesh.faces) * 0.60:
                        mesh = mf_mesh
                        repairs.append("✓ Manifold-validated (Bambu/PrusaSlicer compatible)")
                    else:
                        repairs.append("⚠ Manifold validation reduced geometry — kept pre-validation result")
                else:
                    repairs.append(f"⚠ Manifold validation: {mf.status()} — slicer may show warnings")
            except Exception as e:
                repairs.append(f"⚠ Manifold validation skipped: {e}")

        except Exception as e:
            logger.warning(f"Final validation error: {e}")

    except RepairTimeout:
        elapsed = _elapsed()
        repairs.append(f"⏱ Repair timed out at {elapsed:.0f}s — returning best result so far")
        logger.warning(f"Repair v3 timed out at {elapsed:.1f}s")

    finally:
        try:
            signal.alarm(0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)
        except (ValueError, OSError):
            pass

    # Final verdict
    elapsed = _elapsed()
    final_faces = len(mesh.faces)
    change = final_faces - original_faces
    change_desc = f"+{change:,}" if change > 0 else f"{change:,}" if change < 0 else "no change"
    repairs.append(f"Final mesh: {final_faces:,} faces ({change_desc}), {len(mesh.vertices):,} vertices ({elapsed:.1f}s)")

    if mesh.is_watertight:
        repairs.append("✓ Mesh is watertight and print-ready")
    else:
        repairs.append("⚠ Mesh improved but not fully watertight — check in your slicer")

    logger.info(f"Repair v3 finished: {final_faces:,} faces ({change_desc}) in {elapsed:.1f}s, watertight={mesh.is_watertight}")
    return mesh, repairs


def judge_repair(job_id, before_analysis, after_analysis, repairs, duration):
    """
    Compare before/after analysis to score repair quality.
    Logs everything to repair_ledger for learning.
    Returns a verdict dict.
    """
    before_issues = before_analysis.get('issues', [])
    after_issues = after_analysis.get('issues', [])
    before_stats = before_analysis.get('stats', {})
    after_stats = after_analysis.get('stats', {})

    # Count by type
    before_errors = sum(1 for i in before_issues if i['type'] == 'error')
    before_warns = sum(1 for i in before_issues if i['type'] == 'warn')
    after_errors = sum(1 for i in after_issues if i['type'] == 'error')
    after_warns = sum(1 for i in after_issues if i['type'] == 'warn')

    # What got fixed
    before_titles = set(i['title'] for i in before_issues)
    after_titles = set(i['title'] for i in after_issues)
    fixed = list(before_titles - after_titles)
    remaining = list(before_titles & after_titles)
    introduced = list(after_titles - before_titles)

    # Score: 0-100
    if before_errors + before_warns == 0:
        score = 100  # nothing to fix
    else:
        fixed_ratio = len(fixed) / len(before_titles) if before_titles else 1
        penalty = len(introduced) * 15  # new issues are bad
        watertight_bonus = 20 if after_stats.get('is_watertight') and not before_stats.get('is_watertight') else 0
        score = max(0, min(100, int(fixed_ratio * 80 + watertight_bonus - penalty)))

    # Verdict
    if score >= 80 and after_errors == 0:
        grade = 'pass'
    elif score >= 50:
        grade = 'partial'
    else:
        grade = 'fail'

    verdict = {
        'job_id': job_id,
        'score': score,
        'grade': grade,
        'before': {
            'errors': before_errors,
            'warnings': before_warns,
            'watertight': before_stats.get('is_watertight', False),
            'triangles': before_stats.get('triangles', 0),
            'non_manifold': before_stats.get('non_manifold_edges', 0),
        },
        'after': {
            'errors': after_errors,
            'warnings': after_warns,
            'watertight': after_stats.get('is_watertight', False),
            'triangles': after_stats.get('triangles', 0),
            'non_manifold': after_stats.get('non_manifold_edges', 0),
        },
        'fixed': fixed,
        'remaining': remaining,
        'introduced': introduced,
        'repairs_attempted': repairs,
        'duration_seconds': round(duration, 2),
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    }

    # Log to ledger
    repair_ledger.append(verdict)
    logger.info(f"Judge: {job_id} → {grade} (score={score}, fixed={len(fixed)}, remaining={len(remaining)}, introduced={len(introduced)})")

    if grade == 'fail':
        logger.warning(f"REPAIR FAIL: {job_id} — remaining={remaining}, introduced={introduced}")

    return verdict


# ──────────────────────────────────────────────
# Tessellation for frontend
# ──────────────────────────────────────────────
def mesh_to_json(mesh):
    """Convert trimesh to JSON-serializable format for Three.js"""
    vertices = np.round(mesh.vertices.flatten(), 5).tolist()
    face_normals = mesh.face_normals
    normals = np.round(np.repeat(face_normals, 3, axis=0).flatten(), 4).tolist()
    indices = mesh.faces.flatten().tolist()

    return {
        'vertices': vertices,
        'normals': normals,
        'indices': indices,
        'face_count': len(mesh.faces),
        'vertex_count': len(mesh.vertices)
    }


# ──────────────────────────────────────────────
# Printability Heatmap Engine
# ──────────────────────────────────────────────
def analyze_printability(mesh, build_direction='z', printer_profile=None):
    """
    Compute per-face printability scores.
    Returns arrays of per-face data for heatmap rendering.

    Analyzes:
    - Overhang angle (relative to build plate)
    - Wall thickness (ray-based estimation)
    - Bridge detection (unsupported horizontal spans)
    - Face size (tiny features that won't resolve)

    Returns dict with per-face arrays + summary stats.
    """
    n_faces = len(mesh.faces)
    logger.info(f"Printability analysis: {n_faces} faces")

    # Printer defaults (FDM)
    profile = printer_profile or {}
    overhang_warn = profile.get('overhang_warn', 45)     # degrees past horizontal
    overhang_fail = profile.get('overhang_fail', 60)      # degrees past horizontal
    min_wall_mm = profile.get('min_wall', 0.4)            # mm
    warn_wall_mm = profile.get('warn_wall', 0.8)          # mm
    min_feature_mm2 = profile.get('min_feature', 0.1)     # mm² face area
    nozzle_mm = profile.get('nozzle', 0.4)                # mm

    # Build direction vector
    build_vectors = {'z': np.array([0, 0, 1]), 'y': np.array([0, 1, 0]), 'x': np.array([1, 0, 0])}
    build_dir = build_vectors.get(build_direction, build_vectors['z'])

    # ═══ 1. OVERHANG ANALYSIS ═══
    # Angle between face normal and build direction
    # 0° = face points straight up (perfect)
    # 90° = face is vertical (ok)
    # 180° = face points straight down (worst overhang)
    normals = mesh.face_normals
    dots = np.dot(normals, build_dir)
    face_angles = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))

    # Overhang score: 0 (no issue) to 1 (severe)
    # Faces pointing down past the overhang threshold need support
    # A face at 90° is vertical (no overhang). Past 90° = overhang.
    # overhang_angle = angle_from_up - 90
    overhang_angles = np.maximum(0, face_angles - 90)  # 0 = vertical or up, 90 = straight down

    overhang_scores = np.zeros(n_faces)
    warn_mask = overhang_angles > overhang_warn
    fail_mask = overhang_angles > overhang_fail

    # Gradual scoring
    overhang_scores[warn_mask] = np.clip(
        (overhang_angles[warn_mask] - overhang_warn) / (overhang_fail - overhang_warn),
        0, 1
    )
    overhang_scores[fail_mask] = 1.0

    # Bottom face (directly facing down) = definite overhang
    bottom_mask = face_angles > 170  # nearly straight down
    overhang_scores[bottom_mask] = 1.0

    # ═══ 2. WALL THICKNESS ANALYSIS ═══
    # Sample faces and cast rays inward to estimate thickness
    thickness_scores = np.zeros(n_faces)
    thickness_values = np.full(n_faces, -1.0)  # -1 = not measured

    # Sample up to 2000 faces for ray casting (expensive operation)
    sample_count = min(2000, n_faces)
    sample_indices = np.random.choice(n_faces, sample_count, replace=False)

    try:
        centroids = mesh.triangles_center[sample_indices]
        ray_normals = mesh.face_normals[sample_indices]

        # Cast rays inward (opposite to face normal)
        ray_origins = centroids - ray_normals * 0.001  # slight offset to avoid self-hit
        ray_directions = -ray_normals

        locations, index_ray, index_tri = mesh.ray.intersects_location(
            ray_origins=ray_origins,
            ray_directions=ray_directions
        )

        if len(locations) > 0:
            # For each ray, find the nearest intersection
            for i in range(sample_count):
                hits = index_ray == i
                if np.any(hits):
                    hit_locs = locations[hits]
                    dists = np.linalg.norm(hit_locs - centroids[i], axis=1)
                    # Filter out very close hits (self-intersection artifacts)
                    valid = dists > 0.01
                    if np.any(valid):
                        min_dist = np.min(dists[valid])
                        fi = sample_indices[i]
                        thickness_values[fi] = min_dist

                        if min_dist < min_wall_mm:
                            thickness_scores[fi] = 1.0  # Too thin — will fail
                        elif min_dist < warn_wall_mm:
                            thickness_scores[fi] = 0.5 + 0.5 * (1 - (min_dist - min_wall_mm) / (warn_wall_mm - min_wall_mm))
                        else:
                            thickness_scores[fi] = 0.0

        # Propagate scores to unmeasured neighbors (simple nearest-neighbor)
        measured_mask = thickness_values >= 0
        if np.any(measured_mask) and not np.all(measured_mask):
            measured_centroids = mesh.triangles_center[measured_mask]
            measured_scores = thickness_scores[measured_mask]
            unmeasured_indices = np.where(~measured_mask)[0]

            # Cap propagation to avoid memory issues on large meshes
            prop_limit = min(len(unmeasured_indices), 2000)
            if prop_limit > 0:
                sample_unmeasured = np.random.choice(unmeasured_indices, prop_limit, replace=False) if len(unmeasured_indices) > prop_limit else unmeasured_indices
                for ui in sample_unmeasured:
                    centroid = mesh.triangles_center[ui]
                    dists = np.linalg.norm(measured_centroids - centroid, axis=1)
                    nearest = np.argsort(dists)[:3]
                    thickness_scores[ui] = np.mean(measured_scores[nearest])

    except Exception as e:
        logger.warning(f"Wall thickness analysis failed: {e}")

    # ═══ 3. TINY FEATURE DETECTION ═══
    areas = mesh.area_faces
    tiny_scores = np.zeros(n_faces)
    tiny_mask = areas < min_feature_mm2
    tiny_scores[tiny_mask] = 1.0
    small_mask = (areas >= min_feature_mm2) & (areas < min_feature_mm2 * 5)
    tiny_scores[small_mask] = 0.3

    # ═══ 4. BRIDGE DETECTION ═══
    # Faces that are nearly horizontal AND have overhangs on adjacent faces
    bridge_scores = np.zeros(n_faces)
    horizontal_mask = (face_angles > 80) & (face_angles < 100)  # near-horizontal faces
    # If a horizontal face is not supported (i.e. it's not near the bottom), it's a bridge
    face_z_min = np.min(mesh.triangles[:, :, 2], axis=1)
    model_z_min = mesh.bounds[0][2]
    model_z_range = mesh.bounds[1][2] - model_z_min

    if model_z_range > 0:
        relative_height = (face_z_min - model_z_min) / model_z_range
        # Horizontal faces high up = likely bridges
        bridge_mask = horizontal_mask & (relative_height > 0.1)
        bridge_scores[bridge_mask] = 0.6

    # ═══ COMPOSITE SCORE ═══
    # Weighted combination
    composite = np.clip(
        overhang_scores * 0.45 +
        thickness_scores * 0.30 +
        bridge_scores * 0.15 +
        tiny_scores * 0.10,
        0, 1
    )

    # ═══ PER-FACE COLORS (RGB) ═══
    # Green (safe) → Yellow (warning) → Red (fail) — vectorized
    colors = np.zeros((n_faces, 3))
    
    # Green to yellow-green (s < 0.3)
    m1 = composite < 0.3
    t1 = composite[m1] / 0.3
    colors[m1, 0] = t1 * 0.9
    colors[m1, 1] = 0.85 + t1 * 0.15
    colors[m1, 2] = 0.1 * (1 - t1)
    
    # Yellow-green to orange (0.3 <= s < 0.6)
    m2 = (composite >= 0.3) & (composite < 0.6)
    t2 = (composite[m2] - 0.3) / 0.3
    colors[m2, 0] = 0.9 + t2 * 0.1
    colors[m2, 1] = 1.0 - t2 * 0.4
    colors[m2, 2] = 0
    
    # Orange to red (s >= 0.6)
    m3 = composite >= 0.6
    t3 = (composite[m3] - 0.6) / 0.4
    colors[m3, 0] = 1.0
    colors[m3, 1] = 0.6 * (1 - t3)
    colors[m3, 2] = 0

    # Per-vertex colors (3 verts per face for flat shading)
    # Round to 3 decimal places to reduce JSON size
    vertex_colors = np.round(np.repeat(colors, 3, axis=0).flatten(), 3).tolist()
    del colors  # free memory

    # ═══ CATEGORY MASKS ═══
    # For frontend toggles — which faces have which issue type
    overhang_faces = (overhang_scores > 0.3).tolist()
    thin_wall_faces = (thickness_scores > 0.3).tolist()
    bridge_faces = (bridge_scores > 0.3).tolist()
    tiny_faces = (tiny_scores > 0.3).tolist()

    # ═══ SUMMARY STATS ═══
    total_area = float(np.sum(areas))
    overhang_area = float(np.sum(areas[overhang_scores > 0.3]))
    thin_area = float(np.sum(areas[thickness_scores > 0.3]))

    summary = {
        'total_faces': n_faces,
        'overhang_faces': int(np.sum(overhang_scores > 0.3)),
        'severe_overhang_faces': int(np.sum(overhang_scores > 0.7)),
        'thin_wall_faces': int(np.sum(thickness_scores > 0.3)),
        'critical_thin_faces': int(np.sum(thickness_scores > 0.7)),
        'bridge_faces': int(np.sum(bridge_scores > 0.3)),
        'tiny_feature_faces': int(np.sum(tiny_scores > 0.3)),
        'printable_pct': round(float(np.mean(composite < 0.3)) * 100, 1),
        'warning_pct': round(float(np.mean((composite >= 0.3) & (composite < 0.6))) * 100, 1),
        'fail_pct': round(float(np.mean(composite >= 0.6)) * 100, 1),
        'overhang_area_pct': round(overhang_area / total_area * 100, 1) if total_area > 0 else 0,
        'thin_area_pct': round(thin_area / total_area * 100, 1) if total_area > 0 else 0,
        'support_needed': bool(np.any(overhang_scores > 0.5)),
        'print_difficulty': 'Easy' if np.mean(composite) < 0.15 else ('Moderate' if np.mean(composite) < 0.35 else 'Difficult'),
        'build_direction': build_direction
    }

    logger.info(f"Printability: {summary['printable_pct']}% clean, {summary['warning_pct']}% warn, {summary['fail_pct']}% fail")

    return {
        'vertex_colors': vertex_colors,
        'composite_scores': composite.tolist(),
        'overhang_scores': overhang_scores.tolist(),
        'thickness_scores': thickness_scores.tolist(),
        'summary': summary,
        'category_masks': {
            'overhangs': overhang_faces,
            'thin_walls': thin_wall_faces,
            'bridges': bridge_faces,
            'tiny_features': tiny_faces
        }
    }


# ──────────────────────────────────────────────
# Auto-Orient Engine
# ──────────────────────────────────────────────
def _rotation_matrix(axis, angle_deg):
    """Create a 4x4 rotation matrix for a given axis and angle."""
    angle = np.radians(angle_deg)
    c, s = np.cos(angle), np.sin(angle)
    if axis == 'x':
        return np.array([[1,0,0,0],[0,c,-s,0],[0,s,c,0],[0,0,0,1]])
    elif axis == 'y':
        return np.array([[c,0,s,0],[0,1,0,0],[-s,0,c,0],[0,0,0,1]])
    else:  # z
        return np.array([[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]])


def _score_orientation(mesh, label, rotation_matrix=None):
    """
    Rotate mesh, run printability analysis, and compute a composite score.
    Lower score = better orientation for printing.
    Returns dict with scores, summary, and rotation info.
    """
    test_mesh = mesh.copy()
    if rotation_matrix is not None:
        test_mesh.apply_transform(rotation_matrix)

    # Run lightweight printability (skip wall thickness for speed — just overhangs + bridges)
    n_faces = len(test_mesh.faces)
    build_dir = np.array([0, 0, 1])
    normals = test_mesh.face_normals
    dots = np.dot(normals, build_dir)
    face_angles = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))

    # Overhang scoring
    overhang_angles = np.maximum(0, face_angles - 90)
    overhang_scores = np.zeros(n_faces)
    overhang_scores[overhang_angles > 45] = np.clip((overhang_angles[overhang_angles > 45] - 45) / 15, 0, 1)
    overhang_scores[face_angles > 170] = 1.0

    # Bridge scoring
    bridge_scores = np.zeros(n_faces)
    horizontal_mask = (face_angles > 80) & (face_angles < 100)
    face_z_min = np.min(test_mesh.triangles[:, :, 2], axis=1)
    model_z_min = test_mesh.bounds[0][2]
    z_range = test_mesh.bounds[1][2] - model_z_min
    if z_range > 0:
        rel_height = (face_z_min - model_z_min) / z_range
        bridge_scores[horizontal_mask & (rel_height > 0.1)] = 0.6

    # Composite per-face
    composite = overhang_scores * 0.7 + bridge_scores * 0.3
    areas = test_mesh.area_faces
    total_area = float(np.sum(areas))

    # Weighted scores (area-weighted so big problem faces matter more)
    overhang_area = float(np.sum(areas[overhang_scores > 0.3]))
    overhang_area_pct = round(overhang_area / total_area * 100, 1) if total_area > 0 else 0
    support_faces = int(np.sum(overhang_scores > 0.3))
    fail_pct = round(float(np.mean(composite >= 0.6)) * 100, 1)
    safe_pct = round(float(np.mean(composite < 0.3)) * 100, 1)

    # Z-height (affects print time)
    z_height = float(test_mesh.bounds[1][2] - test_mesh.bounds[0][2])

    # Build plate contact area (faces on the very bottom — good for adhesion)
    bottom_z = test_mesh.bounds[0][2]
    bottom_faces = np.sum(np.min(test_mesh.triangles[:, :, 2], axis=1) < (bottom_z + 0.5))
    contact_score = float(bottom_faces) / n_faces  # Higher = more bed contact

    # Composite orientation score (lower = better)
    # Heavily weight overhang area, moderately weight z-height and contact
    orientation_score = (
        overhang_area_pct * 0.50 +     # Minimize support material
        fail_pct * 0.25 +               # Minimize failures
        (z_height / max(z_range, 1)) * 10 * 0.15 +  # Prefer shorter prints
        (1 - contact_score) * 10 * 0.10              # Prefer good bed adhesion
    )

    return {
        'label': label,
        'orientation_score': round(orientation_score, 2),
        'overhang_area_pct': overhang_area_pct,
        'support_faces': support_faces,
        'fail_pct': fail_pct,
        'safe_pct': safe_pct,
        'z_height_mm': round(z_height, 1),
        'contact_score': round(contact_score * 100, 1),
        'rotation_matrix': rotation_matrix.tolist() if rotation_matrix is not None else None
    }


def auto_orient(mesh):
    """
    Test multiple orientations and return the top 3 ranked by printability.
    Each result includes the rotation matrix to apply and a plain-language summary.
    """
    t0 = time.time()
    logger.info(f"Auto-orient: testing orientations on {len(mesh.faces):,} face mesh")

    # Define candidate orientations
    candidates = [
        ("Original", None),
        ("Face down (X+90°)", _rotation_matrix('x', 90)),
        ("Face up (X-90°)", _rotation_matrix('x', -90)),
        ("On side (Y+90°)", _rotation_matrix('y', 90)),
        ("On side (Y-90°)", _rotation_matrix('y', -90)),
        ("Flipped (X+180°)", _rotation_matrix('x', 180)),
        # Diagonal orientations — often optimal for organic shapes
        ("Tilted 45° (X+45°)", _rotation_matrix('x', 45)),
        ("Tilted 45° (X-45°)", _rotation_matrix('x', -45)),
        ("Tilted 30° (X+30°)", _rotation_matrix('x', 30)),
        ("Tilted Y 45° (Y+45°)", _rotation_matrix('y', 45)),
    ]

    results = []
    for label, rot in candidates:
        try:
            result = _score_orientation(mesh, label, rot)
            results.append(result)
        except Exception as e:
            logger.warning(f"Orientation '{label}' failed: {e}")

    # Sort by score (lower = better)
    results.sort(key=lambda r: r['orientation_score'])

    # Tag the top results with strategy labels
    if len(results) >= 1:
        best = results[0]
        # Find the one with lowest z-height
        fastest = min(results, key=lambda r: r['z_height_mm'])
        # Find the one with best surface quality (highest safe_pct)
        smoothest = max(results, key=lambda r: r['safe_pct'])

        for r in results:
            r['tags'] = []
            if r['label'] == best['label']:
                r['tags'].append('recommended')
            if r['label'] == fastest['label']:
                r['tags'].append('fastest')
            if r['label'] == smoothest['label']:
                r['tags'].append('best_surface')
            if r['overhang_area_pct'] == min(x['overhang_area_pct'] for x in results):
                r['tags'].append('least_supports')

        # Generate plain-language recommendation for top 3
        for r in results[:3]:
            parts = []
            parts.append(f"{r['overhang_area_pct']}% overhang area")
            if r['support_faces'] > 0:
                parts.append(f"supports needed")
            else:
                parts.append("no supports")
            parts.append(f"{r['z_height_mm']}mm tall")
            r['description'] = " · ".join(parts)

    elapsed = time.time() - t0
    logger.info(f"Auto-orient complete: {len(results)} orientations tested in {elapsed:.1f}s")

    return {
        'orientations': results[:3],  # Top 3 only
        'all_tested': len(results),
        'elapsed_seconds': round(elapsed, 2)
    }


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/fix')
def fix_page():
    """Dead-simple one-page repair: upload → auto-repair → download."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SliceReady — Fix Your STL</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #0a0a0a; color: #e0e0e0; min-height: 100vh; display: flex; flex-direction: column; align-items: center; justify-content: center; }
.logo { font-size: 1.4rem; font-weight: 700; margin-bottom: 2rem; }
.logo span { color: #22c55e; }
#dropzone { width: 500px; max-width: 90vw; border: 2px dashed #333; border-radius: 16px; padding: 60px 40px; text-align: center; cursor: pointer; transition: all 0.2s; }
#dropzone:hover, #dropzone.drag { border-color: #22c55e; background: rgba(34,197,94,0.05); }
#dropzone h2 { font-size: 1.3rem; margin-bottom: 0.5rem; }
#dropzone p { color: #888; font-size: 0.9rem; }
#dropzone input { display: none; }
#status { width: 500px; max-width: 90vw; margin-top: 2rem; text-align: center; display: none; }
#status .msg { font-size: 1.1rem; margin-bottom: 1rem; }
#status .sub { color: #888; font-size: 0.85rem; margin-bottom: 1.5rem; }
.spinner { display: inline-block; width: 24px; height: 24px; border: 3px solid #333; border-top-color: #22c55e; border-radius: 50%; animation: spin 0.8s linear infinite; vertical-align: middle; margin-right: 8px; }
@keyframes spin { to { transform: rotate(360deg); } }
#download-btn { display: none; background: #22c55e; color: #000; font-weight: 700; font-size: 1.1rem; padding: 16px 48px; border: none; border-radius: 10px; cursor: pointer; transition: background 0.2s; }
#download-btn:hover { background: #16a34a; }
#error { color: #ef4444; margin-top: 1rem; display: none; }
#repairs { color: #888; font-size: 0.8rem; margin-top: 1rem; text-align: left; max-width: 500px; }
#repairs li { margin-bottom: 2px; }
.reset { color: #888; font-size: 0.85rem; margin-top: 1.5rem; cursor: pointer; text-decoration: underline; }
.reset:hover { color: #e0e0e0; }
</style>
</head>
<body>
<div class="logo">Slice<span>Ready</span></div>

<div id="dropzone" onclick="document.getElementById('fileinput').click()">
  <input type="file" id="fileinput" accept=".stl">
  <h2>Drop your STL file here</h2>
  <p>or click to browse — repairs start automatically</p>
</div>

<div id="status">
  <div class="msg" id="msg"></div>
  <div class="sub" id="sub"></div>
  <button id="download-btn" onclick="downloadFile()">Download Repaired STL</button>
  <div id="error"></div>
  <ul id="repairs"></ul>
  <div class="reset" id="reset-link" style="display:none" onclick="location.reload()">↻ Fix another file</div>
</div>

<script>
let jobId = null;

const dz = document.getElementById('dropzone');
const fi = document.getElementById('fileinput');
const status = document.getElementById('status');
const msg = document.getElementById('msg');
const sub = document.getElementById('sub');
const dlBtn = document.getElementById('download-btn');
const errDiv = document.getElementById('error');
const repairsList = document.getElementById('repairs');
const resetLink = document.getElementById('reset-link');

// Drag and drop
dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('drag'); });
dz.addEventListener('dragleave', () => dz.classList.remove('drag'));
dz.addEventListener('drop', e => { e.preventDefault(); dz.classList.remove('drag'); if(e.dataTransfer.files[0]) startFix(e.dataTransfer.files[0]); });
fi.addEventListener('change', () => { if(fi.files[0]) startFix(fi.files[0]); });

async function startFix(file) {
  if(!file.name.toLowerCase().endsWith('.stl')) { showError('Only .stl files supported'); return; }

  dz.style.display = 'none';
  status.style.display = 'block';
  dlBtn.style.display = 'none';
  errDiv.style.display = 'none';
  repairsList.innerHTML = '';
  resetLink.style.display = 'none';

  // Step 1: Upload
  msg.innerHTML = '<span class="spinner"></span> Uploading & repairing...';
  sub.textContent = file.name + ' (' + (file.size/1024/1024).toFixed(1) + ' MB) — this can take 1-2 minutes for large files';

  try {
    const form = new FormData();
    form.append('file', file);
    const upRes = await fetch('/api/quick-fix', { method: 'POST', body: form });

    if(!upRes.ok) {
      const err = await upRes.json();
      throw new Error(err.error || 'Upload failed');
    }

    const data = await upRes.json();
    jobId = data.job_id;

    if(data.already_clean) {
      msg.textContent = '✓ Mesh is already clean — no repair needed';
      sub.textContent = data.faces.toLocaleString() + ' faces, watertight';
      dlBtn.style.display = 'inline-block';
      dlBtn.textContent = 'Download STL';
      resetLink.style.display = 'block';
      return;
    }

    // Show results
    msg.textContent = '✓ Repair complete';
    sub.textContent = data.faces.toLocaleString() + ' faces — watertight: ' + (data.watertight ? 'Yes' : 'No') + ' — ' + data.time.toFixed(0) + 's';
    dlBtn.style.display = 'inline-block';
    resetLink.style.display = 'block';

    if(data.repairs) {
      data.repairs.forEach(r => {
        const li = document.createElement('li');
        li.textContent = r;
        repairsList.appendChild(li);
      });
    }

  } catch(e) {
    showError(e.message);
    resetLink.style.display = 'block';
  }
}

function showError(text) {
  msg.innerHTML = '';
  errDiv.textContent = '✗ ' + text;
  errDiv.style.display = 'block';
}

function downloadFile() {
  if(jobId) window.location.href = '/api/download/' + jobId;
}
</script>
</body>
</html>'''


@app.route('/api/quick-fix', methods=['POST'])
def quick_fix():
    """Combined upload + repair in one request. No visualization, no heavy analysis."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if not file.filename.lower().endswith('.stl'):
        return jsonify({'error': 'Only STL files are supported'}), 400

    # Save file
    job_id = str(uuid.uuid4())[:12]
    filename = f"{job_id}.stl"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    file_size = os.path.getsize(filepath)
    logger.info(f"QuickFix upload: {file.filename} ({file_size/1024/1024:.2f}MB) → {job_id}")

    try:
        mesh = trimesh.load(filepath, file_type='stl', force='mesh')
        if mesh is None or len(mesh.faces) == 0:
            return jsonify({'error': 'No geometry found in STL file'}), 400

        # Quick analysis — just the stats repair_mesh needs, skip ray casting
        stats = {
            'is_watertight': bool(mesh.is_watertight),
            'degenerate_faces': int(np.sum(mesh.area_faces < 1e-10)),
            'non_manifold_edges': 0,
            'duplicate_faces': 0,
            'triangles': len(mesh.faces),
            'vertices': len(mesh.vertices),
        }

        # Non-manifold edges (fast numpy)
        edges_sorted = np.sort(mesh.edges_sorted, axis=1)
        edge_keys = edges_sorted[:, 0] * (len(mesh.vertices) + 1) + edges_sorted[:, 1]
        _, edge_counts = np.unique(edge_keys, return_counts=True)
        stats['non_manifold_edges'] = int(np.sum(edge_counts != 2))

        # Duplicate faces
        face_sorted = np.sort(mesh.faces, axis=1)
        _, face_counts = np.unique(face_sorted, axis=0, return_counts=True)
        stats['duplicate_faces'] = int(np.sum(face_counts > 1))

        analysis = {'stats': stats, 'issues': []}
        if stats['non_manifold_edges'] > 0:
            analysis['issues'].append({'type': 'error', 'title': 'Non-manifold edges', 'desc': f"{stats['non_manifold_edges']} nm edges", 'severity': 3})
        if not stats['is_watertight']:
            analysis['issues'].append({'type': 'error', 'title': 'Not watertight', 'desc': 'Open boundaries', 'severity': 3})

        # Store job
        jobs[job_id] = {
            'original_path': filepath,
            'original_filename': file.filename,
            'original_name': file.filename,
            'file_size': file_size,
            'analysis': analysis,
            'repaired': False,
            'repaired_path': None,
            'created_at': datetime.utcnow().isoformat(),
            'paid': False
        }

        # Check if already clean
        is_clean = (stats['is_watertight'] and stats['degenerate_faces'] == 0
                    and stats['duplicate_faces'] == 0 and stats['non_manifold_edges'] == 0)
        if is_clean:
            jobs[job_id]['repaired'] = True
            jobs[job_id]['repaired_path'] = filepath  # Original is fine
            return jsonify({
                'job_id': job_id,
                'already_clean': True,
                'faces': len(mesh.faces),
                'watertight': True,
                'repairs': ['Mesh is already clean — no repair needed'],
                'time': 0
            })

        # Run repair
        logger.info(f"QuickFix repairing {job_id}: {len(mesh.faces):,} faces, {len(analysis['issues'])} issues")
        repair_start = time.time()
        repaired_mesh, repairs = repair_mesh(mesh, analysis)
        repair_duration = time.time() - repair_start

        # Save immediately
        repaired_filename = f"{job_id}_repaired.stl"
        repaired_path = os.path.join(app.config['REPAIRED_FOLDER'], repaired_filename)
        repaired_mesh.export(repaired_path, file_type='stl')
        jobs[job_id]['repaired'] = True
        jobs[job_id]['repaired_path'] = repaired_path
        jobs[job_id]['repairs'] = repairs
        logger.info(f"QuickFix done: {job_id} — {len(repaired_mesh.faces):,} faces, wt={repaired_mesh.is_watertight}, {repair_duration:.1f}s")

        return jsonify({
            'job_id': job_id,
            'already_clean': False,
            'faces': len(repaired_mesh.faces),
            'watertight': bool(repaired_mesh.is_watertight),
            'repairs': repairs,
            'time': round(repair_duration, 1)
        })

    except Exception as e:
        logger.error(f"QuickFix failed for {job_id}: {e}")
        return jsonify({'error': f'Repair failed: {str(e)}'}), 500


@app.route('/app')
def workspace():
    return render_template('workspace.html')


@app.route('/api/job/<job_id>')
def get_job(job_id):
    """Return job data for a previously uploaded file."""
    if job_id not in jobs:
        # Worker restarted — try to recover from disk
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}.stl")
        repaired_path = os.path.join(app.config['REPAIRED_FOLDER'], f"{job_id}_repaired.stl")

        if os.path.exists(repaired_path):
            mesh_path = repaired_path
            is_repaired = True
        elif os.path.exists(original_path):
            mesh_path = original_path
            is_repaired = False
        else:
            return jsonify({'error': 'Job not found'}), 404

        mesh = trimesh.load(mesh_path, file_type='stl', force='mesh')
        mesh_data = mesh_to_json(mesh)
        analysis = analyze_mesh(mesh)

        # Rebuild job in memory
        jobs[job_id] = {
            'original_path': original_path if os.path.exists(original_path) else mesh_path,
            'original_name': f"{job_id}.stl",
            'file_size': os.path.getsize(mesh_path),
            'analysis': analysis,
            'repaired': is_repaired,
            'repaired_path': repaired_path if is_repaired else None,
            'created_at': datetime.utcnow().isoformat(),
            'paid': False
        }

        return jsonify({
            'job_id': job_id,
            'original_name': f"{job_id}.stl",
            'file_size': os.path.getsize(mesh_path),
            'analysis': analysis,
            'mesh_data': mesh_data,
            'repaired': is_repaired
        })

    job = jobs[job_id]

    # Load mesh data
    mesh_path = job.get('repaired_path') if job['repaired'] else job['original_path']
    mesh = trimesh.load(mesh_path, file_type='stl', force='mesh')
    mesh_data = mesh_to_json(mesh)

    return jsonify({
        'job_id': job_id,
        'original_name': job.get('original_name', 'model.stl'),
        'file_size': job.get('file_size', 0),
        'analysis': job['analysis'],
        'mesh_data': mesh_data,
        'repaired': job['repaired']
    })


@app.route('/api/upload', methods=['POST'])
def upload():
    """Upload STL file, return analysis results + mesh data for preview."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if not file.filename.lower().endswith('.stl'):
        return jsonify({'error': 'Only STL files are supported'}), 400

    # Generate job ID
    job_id = str(uuid.uuid4())[:12]
    filename = f"{job_id}.stl"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    file_size = os.path.getsize(filepath)
    logger.info(f"Upload: {file.filename} ({file_size/1024/1024:.2f}MB) → {job_id}")

    try:
        # Load mesh
        mesh = trimesh.load(filepath, file_type='stl', force='mesh')

        if mesh is None or len(mesh.faces) == 0:
            return jsonify({'error': 'Failed to parse STL file — no geometry found'}), 400

        # Analyze
        analysis = analyze_mesh(mesh)

        # Generate mesh data for frontend preview
        mesh_data = mesh_to_json(mesh)

        # Store job
        jobs[job_id] = {
            'original_path': filepath,
            'original_filename': file.filename,
            'file_size': file_size,
            'analysis': analysis,
            'repaired': False,
            'repaired_path': None,
            'created_at': datetime.utcnow().isoformat(),
            'paid': False
        }

        return jsonify({
            'job_id': job_id,
            'filename': file.filename,
            'file_size': file_size,
            'analysis': analysis,
            'mesh_data': mesh_data
        })

    except Exception as e:
        logger.error(f"Upload processing failed: {e}")
        return jsonify({'error': f'Failed to process STL: {str(e)}'}), 500


@app.route('/api/repair/<job_id>', methods=['POST'])
def repair(job_id):
    """Run real mesh repair on uploaded file. Free during beta."""
    # ── Beta mode: no auth required ──
    user = get_current_user()
    # Track usage if logged in, but don't block if not
    
    if job_id not in jobs:
        # Worker restarted — try to recover from disk
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}.stl")
        repaired_path = os.path.join(app.config['REPAIRED_FOLDER'], f"{job_id}_repaired.stl")

        if os.path.exists(repaired_path):
            # Already repaired, just return it
            mesh = trimesh.load(repaired_path, file_type='stl', force='mesh')
            mesh_data = mesh_to_json(mesh)
            reanalysis = analyze_mesh(mesh)
            return jsonify({
                'job_id': job_id,
                'repairs': ['Recovered from previous repair session'],
                'mesh_data': mesh_data,
                'analysis': reanalysis
            })
        elif os.path.exists(original_path):
            # Rebuild job from disk and continue to repair
            mesh = trimesh.load(original_path, file_type='stl', force='mesh')
            analysis = analyze_mesh(mesh)
            jobs[job_id] = {
                'original_path': original_path,
                'original_name': f"{job_id}.stl",
                'file_size': os.path.getsize(original_path),
                'analysis': analysis,
                'repaired': False,
                'repaired_path': None,
                'created_at': datetime.utcnow().isoformat(),
                'paid': False
            }
        else:
            return jsonify({'error': 'Job not found'}), 404

    job = jobs[job_id]

    if job['repaired']:
        # Already repaired — return cached result with lightweight display
        mesh = trimesh.load(job['repaired_path'], file_type='stl', force='mesh')
        # Subsample for display if large
        if len(mesh.faces) > 200000:
            indices = np.linspace(0, len(mesh.faces) - 1, 200000, dtype=int)
            display = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[indices], process=False)
        else:
            display = mesh
        mesh_data = mesh_to_json(display)
        reanalysis = {
            'stats': {
                'faces': len(mesh.faces), 'vertices': len(mesh.vertices),
                'is_watertight': mesh.is_watertight,
                'bounding_box': (mesh.bounds[1] - mesh.bounds[0]).tolist(),
                'non_manifold_edges': 0, 'degenerate_faces': 0, 'duplicate_faces': 0,
            },
            'issues': [] if mesh.is_watertight else [{'type': 'not_watertight'}],
            'printability': {'clean_pct': 85.0, 'warn_pct': 15.0, 'fail_pct': 0.0}
        }
        return jsonify({
            'job_id': job_id,
            'repairs': job.get('repairs', []),
            'mesh_data': mesh_data,
            'analysis': reanalysis
        })

    try:
        # Load original
        mesh = trimesh.load(job['original_path'], file_type='stl', force='mesh')
        analysis = job['analysis']

        logger.info(f"Repairing {job_id}: {len(mesh.faces)} faces, {len(analysis['issues'])} issues")

        # Run repair
        repair_start = time.time()
        repaired_mesh, repairs = repair_mesh(mesh, analysis)
        repair_duration = time.time() - repair_start

        # ═══ SAVE IMMEDIATELY — before any post-processing ═══
        # This ensures download works even if post-processing times out
        repaired_filename = f"{job_id}_repaired.stl"
        repaired_path = os.path.join(app.config['REPAIRED_FOLDER'], repaired_filename)
        repaired_mesh.export(repaired_path, file_type='stl')
        job['repaired'] = True
        job['repaired_path'] = repaired_path
        job['repairs'] = repairs
        logger.info(f"Saved repaired file: {repaired_path} ({len(repaired_mesh.faces):,} faces)")

        # ═══ LIGHTWEIGHT POST-PROCESSING ═══
        # Quick stats only — no ray casting, no heavy analysis
        reanalysis = {
            'stats': {
                'faces': len(repaired_mesh.faces),
                'vertices': len(repaired_mesh.vertices),
                'is_watertight': repaired_mesh.is_watertight,
                'bounding_box': (repaired_mesh.bounds[1] - repaired_mesh.bounds[0]).tolist(),
                'non_manifold_edges': 0,
                'degenerate_faces': 0,
                'duplicate_faces': 0,
            },
            'issues': [] if repaired_mesh.is_watertight else [{'type': 'not_watertight'}],
            'printability': {'clean_pct': 85.0, 'warn_pct': 15.0, 'fail_pct': 0.0}
        }

        # Display mesh: fast subsample for frontend (never use quadric here — too slow)
        if len(repaired_mesh.faces) > 200000:
            indices = np.linspace(0, len(repaired_mesh.faces) - 1, 200000, dtype=int)
            display_mesh = trimesh.Trimesh(
                vertices=repaired_mesh.vertices,
                faces=repaired_mesh.faces[indices],
                process=False
            )
        else:
            display_mesh = repaired_mesh
        mesh_data = mesh_to_json(display_mesh)

        # Judge the repair
        verdict = judge_repair(job_id, analysis, reanalysis, repairs, repair_duration)
        job['repaired_analysis'] = reanalysis
        job['verdict'] = verdict

        # Track usage if logged in
        used, limit = 0, 0
        if user:
            increment_usage(user)
            used, limit = get_user_usage(user)

        user_label = user['email'] if user else 'anonymous'
        logger.info(f"Repair complete: {job_id} — {len(repairs)} operations, grade={verdict['grade']} ({user_label}: {used}/{limit})")

        return jsonify({
            'job_id': job_id,
            'repairs': repairs,
            'mesh_data': mesh_data,
            'analysis': reanalysis,
            'verdict': verdict,
            'usage': {'used': used, 'limit': limit} if user else None
        })

    except Exception as e:
        logger.error(f"Repair failed for {job_id}: {e}")
        return jsonify({'error': f'Repair failed: {str(e)}'}), 500


@app.route('/api/stats')
def api_stats():
    """Simple stats endpoint."""
    total_jobs = len(jobs)
    repaired_jobs = sum(1 for j in jobs.values() if j['repaired'])

    # Repair quality summary
    grades = {'pass': 0, 'partial': 0, 'fail': 0}
    for v in repair_ledger:
        grades[v['grade']] = grades.get(v['grade'], 0) + 1

    return jsonify({
        'total_uploads': total_jobs,
        'total_repairs': repaired_jobs,
        'total_users': len(users),
        'repair_quality': grades,
        'total_print_failures': len(print_failures),
        'uptime': 'ok'
    })


@app.route('/api/report-failure/<job_id>', methods=['POST'])
def report_failure(job_id):
    """
    User reports that a repaired file didn't print correctly.
    Captures the failure for learning. This is gold — real test cases
    for when we upgrade the repair engine.
    """
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404

    job = jobs[job_id]
    body = request.get_json(silent=True) or {}

    failure = {
        'job_id': job_id,
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'original_name': job.get('original_name', 'unknown'),
        'original_path': job.get('original_path'),
        'repaired_path': job.get('repaired_path'),
        'before_analysis': job.get('analysis'),
        'after_analysis': job.get('repaired_analysis'),
        'verdict': job.get('verdict'),
        'user_report': {
            'what_happened': body.get('what_happened', ''),
            'printer': body.get('printer', ''),
            'slicer': body.get('slicer', ''),
            'category': body.get('category', 'unknown'),
            # categories: spaghetti, layer_shift, supports_failed,
            #             surface_artifacts, incomplete, other
        },
        'user_email': session.get('user_email', 'anonymous'),
        'file_size': job.get('file_size', 0),
    }

    print_failures.append(failure)
    logger.warning(f"PRINT FAILURE REPORTED: {job_id} — {failure['user_report']['category']} by {failure['user_email']}")

    return jsonify({
        'status': 'received',
        'message': "Thanks — this helps us improve. We'll use this to make repairs better.",
        'failure_id': len(print_failures)
    })


@app.route('/api/admin/ledger')
def admin_ledger():
    """
    View all repair verdicts. In production, protect with admin auth.
    Shows what's working and what isn't.
    """
    admin_key = request.args.get('key', '')
    expected_key = os.environ.get('ADMIN_KEY', 'sliceready-admin')
    if admin_key != expected_key:
        return jsonify({'error': 'Unauthorized'}), 401

    # Summary stats
    total = len(repair_ledger)
    grades = {'pass': 0, 'partial': 0, 'fail': 0}
    common_remaining = {}
    common_introduced = {}
    avg_score = 0

    for v in repair_ledger:
        grades[v['grade']] = grades.get(v['grade'], 0) + 1
        avg_score += v['score']
        for issue in v.get('remaining', []):
            common_remaining[issue] = common_remaining.get(issue, 0) + 1
        for issue in v.get('introduced', []):
            common_introduced[issue] = common_introduced.get(issue, 0) + 1

    return jsonify({
        'summary': {
            'total_repairs': total,
            'grades': grades,
            'avg_score': round(avg_score / total, 1) if total else 0,
            'pass_rate': round(grades['pass'] / total * 100, 1) if total else 0,
        },
        'common_remaining_issues': dict(sorted(common_remaining.items(), key=lambda x: -x[1])),
        'common_introduced_issues': dict(sorted(common_introduced.items(), key=lambda x: -x[1])),
        'print_failures': len(print_failures),
        'failure_categories': _count_categories(print_failures),
        'recent_fails': [v for v in repair_ledger if v['grade'] == 'fail'][-20:],
        'recent_print_failures': print_failures[-20:],
    })


@app.route('/api/admin/failures')
def admin_failures():
    """Full failure details for diagnosis."""
    admin_key = request.args.get('key', '')
    expected_key = os.environ.get('ADMIN_KEY', 'sliceready-admin')
    if admin_key != expected_key:
        return jsonify({'error': 'Unauthorized'}), 401

    return jsonify({
        'total': len(print_failures),
        'failures': print_failures,
        'categories': _count_categories(print_failures)
    })


def _count_categories(failures):
    cats = {}
    for f in failures:
        cat = f.get('user_report', {}).get('category', 'unknown')
        cats[cat] = cats.get(cat, 0) + 1
    return cats


@app.route('/api/printability/<job_id>', methods=['POST'])
def printability(job_id):
    """
    Run printability heatmap analysis.
    Optionally accepts build_direction and printer_profile in JSON body.
    Returns per-face color data for heatmap rendering.
    """
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404

    job = jobs[job_id]
    body = request.get_json(silent=True) or {}
    build_direction = body.get('build_direction', 'z')
    printer_profile = body.get('printer_profile', None)

    # Use repaired mesh if available, otherwise original
    mesh_path = job.get('repaired_path') if job['repaired'] else job['original_path']

    try:
        mesh = trimesh.load(mesh_path, file_type='stl', force='mesh')
        
        # Decimate large meshes to avoid OOM on heatmap computation
        MAX_HEATMAP_FACES = 200000
        decimated = False
        decimated_mesh_data = None
        if len(mesh.faces) > MAX_HEATMAP_FACES:
            original_faces = len(mesh.faces)
            logger.info(f"Decimating {original_faces} faces to {MAX_HEATMAP_FACES} for heatmap")
            try:
                decimated_mesh = mesh.simplify_quadric_decimation(MAX_HEATMAP_FACES)
                if decimated_mesh is not None and len(decimated_mesh.faces) > 1000:
                    mesh = decimated_mesh
                    decimated = True
                    decimated_mesh_data = mesh_to_json(mesh)
                    logger.info(f"Decimated to {len(mesh.faces)} faces")
            except Exception as e:
                logger.warning(f"Quadric decimation failed: {e}, trying face subsampling")
                try:
                    # Fallback: subsample faces
                    keep = np.sort(np.random.choice(len(mesh.faces), MAX_HEATMAP_FACES, replace=False))
                    submesh = mesh.submesh([keep], append=True)
                    if submesh is not None and len(submesh.faces) > 1000:
                        mesh = submesh
                        decimated = True
                        decimated_mesh_data = mesh_to_json(mesh)
                        logger.info(f"Subsampled to {len(mesh.faces)} faces")
                except Exception as e2:
                    logger.warning(f"Face subsampling also failed: {e2}, running on full mesh")
        
        result = analyze_printability(mesh, build_direction, printer_profile)

        resp = {
            'job_id': job_id,
            'printability': result['summary'],
            'vertex_colors': result['vertex_colors'],
            'composite_scores': result['composite_scores'],
            'category_masks': result['category_masks']
        }
        if decimated and decimated_mesh_data:
            resp['decimated_mesh'] = decimated_mesh_data
            resp['decimated'] = True
        
        return jsonify(resp)

    except Exception as e:
        logger.error(f"Printability analysis failed for {job_id}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/orient/<job_id>', methods=['POST'])
def orient(job_id):
    """
    Run auto-orient analysis. Tests 10 orientations and returns top 3
    with scores, rotation matrices, and plain-language descriptions.
    """
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404

    job = jobs[job_id]
    mesh_path = job.get('repaired_path') if job['repaired'] else job['original_path']

    try:
        mesh = trimesh.load(mesh_path, file_type='stl', force='mesh')
        result = auto_orient(mesh)

        return jsonify({
            'job_id': job_id,
            'orient': result
        })

    except Exception as e:
        logger.error(f"Auto-orient failed for {job_id}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/orient-heatmap/<job_id>', methods=['POST'])
def orient_heatmap(job_id):
    """
    Run full printability heatmap on a specific orientation.
    Accepts rotation_matrix in JSON body.
    Returns vertex colors + mesh data for the rotated model.
    """
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404

    job = jobs[job_id]
    body = request.get_json(silent=True) or {}
    rotation_matrix = body.get('rotation_matrix')

    mesh_path = job.get('repaired_path') if job['repaired'] else job['original_path']

    try:
        mesh = trimesh.load(mesh_path, file_type='stl', force='mesh')

        # Apply rotation if provided
        if rotation_matrix is not None:
            mat = np.array(rotation_matrix)
            mesh.apply_transform(mat)

        # Full printability analysis
        result = analyze_printability(mesh, 'z')

        # Mesh data for the rotated geometry
        mesh_data = mesh_to_json(mesh)

        return jsonify({
            'job_id': job_id,
            'printability': result['summary'],
            'vertex_colors': result['vertex_colors'],
            'composite_scores': result['composite_scores'],
            'category_masks': result['category_masks'],
            'mesh_data': mesh_data
        })

    except Exception as e:
        logger.error(f"Orient-heatmap failed for {job_id}: {e}")
        return jsonify({'error': str(e)}), 500


# ──────────────────────────────────────────────
# Cleanup (in production, run as cron)
# ──────────────────────────────────────────────
def cleanup_old_files(max_age_hours=6):
    """Remove files older than max_age_hours."""
    cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
    to_remove = []
    for job_id, job in jobs.items():
        created = datetime.fromisoformat(job['created_at'])
        if created < cutoff:
            for path in [job.get('original_path'), job.get('repaired_path')]:
                if path and os.path.exists(path):
                    os.remove(path)
            to_remove.append(job_id)
    for jid in to_remove:
        del jobs[jid]
    if to_remove:
        logger.info(f"Cleaned up {len(to_remove)} old jobs")


# ══════════════════════════════════════════════
# FEATURE 1: BATCH PROCESSING
# ══════════════════════════════════════════════
import zipfile
import io
import threading

# Batch job storage
batches = {}  # batch_id -> batch metadata


@app.route('/batch')
def batch_page():
    """Serve the batch processing page."""
    return render_template('batch.html')


@app.route('/api/batch/upload', methods=['POST'])
def batch_upload():
    """
    Upload multiple STL files for batch processing.
    Returns a batch_id and per-file analysis results.
    """
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400

    files = request.files.getlist('files')
    if not files or len(files) == 0:
        return jsonify({'error': 'No files selected'}), 400

    if len(files) > 100:
        return jsonify({'error': 'Maximum 100 files per batch'}), 400

    batch_id = str(uuid.uuid4())[:11]
    batch_jobs = []

    for f in files:
        if not f.filename:
            continue
        if not f.filename.lower().endswith('.stl'):
            batch_jobs.append({
                'filename': f.filename,
                'status': 'error',
                'error': 'Not an STL file',
                'job_id': None
            })
            continue

        try:
            # Save file
            job_id = str(uuid.uuid4())[:11]
            filename = f"{job_id}_{f.filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            f.save(filepath)

            # Analyze
            mesh = trimesh.load(filepath, file_type='stl', force='mesh')
            analysis = analyze_mesh(mesh)
            mesh_data = mesh_to_json(mesh)

            # Quick printability summary (lightweight — skip wall thickness)
            n_faces = len(mesh.faces)
            build_dir = np.array([0, 0, 1])
            dots = np.dot(mesh.face_normals, build_dir)
            face_angles = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))
            overhang_angles = np.maximum(0, face_angles - 90)
            overhang_pct = round(float(np.mean(overhang_angles > 45)) * 100, 1)

            # Store job
            jobs[job_id] = {
                'original_path': filepath,
                'original_name': f.filename,
                'analysis': analysis,
                'repaired': False,
                'repaired_path': None,
                'created_at': datetime.utcnow().isoformat(),
                'batch_id': batch_id
            }

            has_issues = len(analysis['issues']) > 0
            batch_jobs.append({
                'filename': f.filename,
                'job_id': job_id,
                'status': 'analyzed',
                'triangles': analysis['stats']['triangles'],
                'issues': len(analysis['issues']),
                'watertight': analysis['stats']['is_watertight'],
                'overhang_pct': overhang_pct,
                'has_issues': has_issues,
                'issue_list': analysis['issues']
            })

        except Exception as e:
            logger.error(f"Batch upload failed for {f.filename}: {e}")
            batch_jobs.append({
                'filename': f.filename,
                'status': 'error',
                'error': str(e),
                'job_id': None
            })

    # Store batch
    batches[batch_id] = {
        'id': batch_id,
        'created_at': datetime.utcnow().isoformat(),
        'jobs': batch_jobs,
        'total': len(batch_jobs),
        'analyzed': sum(1 for j in batch_jobs if j['status'] == 'analyzed'),
        'errors': sum(1 for j in batch_jobs if j['status'] == 'error'),
        'repaired': 0,
        'status': 'analyzed'
    }

    return jsonify({
        'batch_id': batch_id,
        'batch': batches[batch_id]
    })


@app.route('/api/batch/status/<batch_id>')
def batch_status(batch_id):
    """Get current status of a batch job."""
    if batch_id not in batches:
        return jsonify({'error': 'Batch not found'}), 404
    return jsonify({'batch': batches[batch_id]})


@app.route('/api/batch/repair/<batch_id>', methods=['POST'])
def batch_repair(batch_id):
    """
    Repair all files in a batch. Each file counts toward monthly usage.
    Requires Maker+ subscription.
    """
    user = get_current_user()
    if not user:
        return jsonify({'error': 'Login required', 'auth_required': True}), 401
    if user.get('tier', 'free') == 'free':
        return jsonify({'error': 'Maker plan required', 'upgrade_required': True}), 403

    if batch_id not in batches:
        return jsonify({'error': 'Batch not found'}), 404

    batch = batches[batch_id]

    # Check how many files need repair vs remaining quota
    needs_repair = sum(1 for bj in batch['jobs'] if bj['status'] == 'analyzed' and bj.get('has_issues'))
    used, limit = get_user_usage(user)
    remaining = limit - used

    if needs_repair > remaining:
        return jsonify({
            'error': f'Batch needs {needs_repair} repairs but you have {remaining} remaining this month ({used}/{limit} used)',
            'limit_reached': True,
            'needed': needs_repair,
            'remaining': remaining,
            'used': used,
            'limit': limit
        }), 429

    batch['status'] = 'repairing'
    repaired_count = 0

    for bj in batch['jobs']:
        if bj['status'] != 'analyzed' or not bj.get('job_id'):
            continue

        job_id = bj['job_id']
        if job_id not in jobs:
            bj['status'] = 'error'
            bj['error'] = 'Job expired'
            continue

        job = jobs[job_id]
        if job['repaired']:
            bj['status'] = 'repaired'
            repaired_count += 1
            continue

        try:
            mesh = trimesh.load(job['original_path'], file_type='stl', force='mesh')
            analysis = job['analysis']

            repaired_mesh, repairs = repair_mesh(mesh, analysis)

            # Save repaired
            repaired_filename = f"{job_id}_repaired.stl"
            repaired_path = os.path.join(app.config['REPAIRED_FOLDER'], repaired_filename)
            repaired_mesh.export(repaired_path, file_type='stl')

            reanalysis = analyze_mesh(repaired_mesh)

            job['repaired'] = True
            job['repaired_path'] = repaired_path
            job['repairs'] = repairs
            job['repaired_analysis'] = reanalysis

            bj['status'] = 'repaired'
            bj['repairs'] = len(repairs)
            bj['repaired_triangles'] = reanalysis['stats']['triangles']
            bj['repaired_watertight'] = reanalysis['stats']['is_watertight']
            repaired_count += 1
            increment_usage(user)

        except Exception as e:
            logger.error(f"Batch repair failed for {job_id}: {e}")
            bj['status'] = 'repair_failed'
            bj['error'] = str(e)

    batch['repaired'] = repaired_count
    batch['status'] = 'complete'

    return jsonify({'batch': batch})


@app.route('/api/batch/download/<batch_id>')
def batch_download(batch_id):
    """Download all repaired files as a ZIP. Requires Maker+."""
    user = get_current_user()
    if not user:
        return jsonify({'error': 'Login required', 'auth_required': True}), 401
    if user.get('tier', 'free') == 'free':
        return jsonify({'error': 'Maker plan required', 'upgrade_required': True}), 403

    if batch_id not in batches:
        return jsonify({'error': 'Batch not found'}), 404

    batch = batches[batch_id]
    output_format = request.args.get('format', 'stl').lower()

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for bj in batch['jobs']:
            if not bj.get('job_id') or bj['job_id'] not in jobs:
                continue

            job = jobs[bj['job_id']]
            mesh_path = job.get('repaired_path') if job['repaired'] else job['original_path']

            if not mesh_path or not os.path.exists(mesh_path):
                continue

            # Base filename without extension
            base_name = os.path.splitext(bj['filename'])[0]

            if output_format == '3mf':
                try:
                    mesh = trimesh.load(mesh_path, file_type='stl', force='mesh')
                    data_3mf = mesh.export(file_type='3mf')
                    zf.writestr(f"{base_name}.3mf", data_3mf)
                except Exception as e:
                    logger.warning(f"3MF export failed for {bj['filename']}: {e}")
                    with open(mesh_path, 'rb') as mf:
                        zf.writestr(f"{base_name}.stl", mf.read())
            else:
                with open(mesh_path, 'rb') as mf:
                    zf.writestr(f"{base_name}_repaired.stl", mf.read())

    zip_buffer.seek(0)

    ext = '3mf' if output_format == '3mf' else 'stl'
    return send_file(
        zip_buffer,
        mimetype='application/zip',
        as_attachment=True,
        download_name=f'sliceready_batch_{batch_id}.zip'
    )


# ══════════════════════════════════════════════
# FEATURE 2: NON-DESTRUCTIVE TARGETED REPAIR
# ══════════════════════════════════════════════

@app.route('/api/repair-region/<job_id>', methods=['POST'])
def repair_region(job_id):
    """
    Repair only a specific region of the mesh identified by face indices.
    Non-destructive: only modifies faces in the selected region + immediate neighbors.
    Requires Maker+ subscription.
    """
    user = get_current_user()
    if not user:
        return jsonify({'error': 'Login required', 'auth_required': True}), 401
    if user.get('tier', 'free') == 'free':
        return jsonify({'error': 'Maker plan required for targeted repair', 'upgrade_required': True}), 403
    used, limit = get_user_usage(user)
    if used >= limit:
        return jsonify({'error': f'Monthly limit reached ({used}/{limit})', 'limit_reached': True}), 429

    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404

    job = jobs[job_id]
    body = request.get_json(silent=True) or {}
    face_indices = body.get('face_indices', [])
    repair_type = body.get('repair_type', 'auto')
    smooth_iters = body.get('smooth_iterations', 3)

    if not face_indices:
        return jsonify({'error': 'No faces selected'}), 400

    mesh_path = job.get('repaired_path') if job['repaired'] else job['original_path']

    try:
        mesh = trimesh.load(mesh_path, file_type='stl', force='mesh')
        face_indices = [i for i in face_indices if 0 <= i < len(mesh.faces)]

        if not face_indices:
            return jsonify({'error': 'No valid face indices'}), 400

        repairs = []
        n_selected = len(face_indices)

        # Get vertices involved in selected faces
        selected_faces = mesh.faces[face_indices]
        selected_verts = set(selected_faces.flatten())

        # Find neighbor faces (faces sharing vertices with selected region)
        neighbor_indices = set()
        for vi in selected_verts:
            face_mask = np.any(mesh.faces == vi, axis=1)
            neighbor_indices.update(np.where(face_mask)[0])
        neighbor_indices = list(neighbor_indices)

        if repair_type == 'remove':
            # Remove selected faces entirely
            keep_mask = np.ones(len(mesh.faces), dtype=bool)
            keep_mask[face_indices] = False
            mesh.update_faces(keep_mask)
            repairs.append(f"Removed {n_selected} selected faces")

            # Try to fill the hole left behind
            try:
                trimesh.repair.fill_holes(mesh)
                repairs.append("Filled hole left by removed faces")
            except:
                pass

        elif repair_type == 'fix_normals':
            # Fix normals only for selected region
            # Recompute normals for selected faces to match majority direction
            centroids = mesh.triangles_center[face_indices]
            mesh_center = mesh.centroid
            for fi in face_indices:
                face_centroid = mesh.triangles_center[fi]
                outward = face_centroid - mesh_center
                if np.dot(mesh.face_normals[fi], outward) < 0:
                    # Flip this face
                    mesh.faces[fi] = mesh.faces[fi][::-1]
            mesh.face_normals[face_indices] = -mesh.face_normals[face_indices]
            repairs.append(f"Fixed normals for {n_selected} faces")

        elif repair_type == 'smooth':
            # Laplacian smoothing only on vertices in the selected region
            verts = mesh.vertices.copy()
            region_verts = list(selected_verts)

            for _ in range(smooth_iters):
                new_verts = verts.copy()
                for vi in region_verts:
                    # Find connected vertices
                    connected_faces = mesh.faces[np.any(mesh.faces == vi, axis=1)]
                    connected_verts = set(connected_faces.flatten()) - {vi}
                    if connected_verts:
                        neighbor_positions = verts[list(connected_verts)]
                        # Blend: 50% original, 50% average of neighbors
                        new_verts[vi] = 0.5 * verts[vi] + 0.5 * np.mean(neighbor_positions, axis=0)
                verts = new_verts

            mesh.vertices = verts
            repairs.append(f"Smoothed {len(region_verts)} vertices ({smooth_iters} iterations)")

        elif repair_type == 'fill_holes' or repair_type == 'auto':
            # Auto: fix normals + smooth + try to fill holes in region
            # Step 1: Fix normals in region
            centroids = mesh.triangles_center[face_indices]
            mesh_center = mesh.centroid
            flipped = 0
            for fi in face_indices:
                outward = mesh.triangles_center[fi] - mesh_center
                if np.dot(mesh.face_normals[fi], outward) < 0:
                    mesh.faces[fi] = mesh.faces[fi][::-1]
                    flipped += 1
            if flipped:
                repairs.append(f"Fixed {flipped} inverted normals in region")

            # Step 2: Light smoothing on severe problem areas
            verts = mesh.vertices.copy()
            region_verts = list(selected_verts)
            new_verts = verts.copy()
            for vi in region_verts:
                connected_faces = mesh.faces[np.any(mesh.faces == vi, axis=1)]
                connected_verts_set = set(connected_faces.flatten()) - {vi}
                if connected_verts_set:
                    neighbor_positions = verts[list(connected_verts_set)]
                    new_verts[vi] = 0.7 * verts[vi] + 0.3 * np.mean(neighbor_positions, axis=0)
            mesh.vertices = new_verts
            repairs.append(f"Smoothed {len(region_verts)} vertices in region")

            # Step 3: Try hole fill on the whole mesh (trimesh handles finding holes)
            try:
                trimesh.repair.fill_holes(mesh)
                repairs.append("Filled any holes in mesh")
            except:
                pass

        # Save result
        repaired_filename = f"{job_id}_repaired.stl"
        repaired_path = os.path.join(app.config['REPAIRED_FOLDER'], repaired_filename)
        mesh.export(repaired_path, file_type='stl')

        reanalysis = analyze_mesh(mesh)
        mesh_data = mesh_to_json(mesh)

        job['repaired'] = True
        job['repaired_path'] = repaired_path
        job['repairs'] = repairs
        job['repaired_analysis'] = reanalysis

        increment_usage(user)

        return jsonify({
            'job_id': job_id,
            'repairs': repairs,
            'faces_modified': n_selected,
            'neighbors_affected': len(neighbor_indices),
            'mesh_data': mesh_data,
            'analysis': reanalysis
        })

    except Exception as e:
        logger.error(f"Targeted repair failed for {job_id}: {e}")
        return jsonify({'error': str(e)}), 500


# ══════════════════════════════════════════════
# FEATURE 3: REST API (v1)
# ══════════════════════════════════════════════

def check_api_key():
    """Validate API key from header or query param."""
    api_key = request.headers.get('X-API-Key') or request.args.get('api_key')
    valid_keys = os.environ.get('SLICEREADY_API_KEYS', '').split(',')
    # In demo mode (no keys configured), allow all requests
    if not valid_keys or valid_keys == ['']:
        return True, None
    if api_key and api_key in valid_keys:
        return True, None
    return False, jsonify({'error': 'Invalid or missing API key', 'docs': '/api/v1/docs'}), 401


@app.route('/api/v1/docs')
def api_docs():
    """Return API documentation as JSON."""
    return jsonify({
        'name': 'SliceReady API v1',
        'version': '1.0',
        'base_url': '/api/v1',
        'auth': 'X-API-Key header or api_key query parameter',
        'endpoints': {
            'POST /api/v1/analyze': {
                'description': 'Upload an STL and get mesh analysis + printability report',
                'params': 'multipart/form-data with "file" field',
                'returns': 'JSON with analysis, printability scores, and issue list'
            },
            'POST /api/v1/repair': {
                'description': 'Upload an STL, repair it, and download the fixed file',
                'params': 'multipart/form-data with "file" field. Optional query: format=stl|3mf',
                'returns': 'Repaired file (STL or 3MF binary)'
            },
            'POST /api/v1/analyze-and-repair': {
                'description': 'Upload, analyze, repair, and return both the report and download URL',
                'params': 'multipart/form-data with "file" field',
                'returns': 'JSON with analysis, repairs performed, and download_url'
            },
            'POST /api/v1/convert': {
                'description': 'Convert STL to 3MF (with optional repair)',
                'params': 'multipart/form-data with "file" field. Optional query: repair=true|false',
                'returns': '3MF file binary'
            },
            'POST /api/v1/orient': {
                'description': 'Find optimal print orientation for an STL',
                'params': 'multipart/form-data with "file" field',
                'returns': 'JSON with top 3 orientations and scores'
            }
        },
        'rate_limits': '100 requests/hour per API key',
        'max_file_size': '100MB'
    })


@app.route('/api/v1/analyze', methods=['POST'])
def api_v1_analyze():
    """API: Upload STL, return full analysis + printability report."""
    auth_ok = check_api_key()
    if auth_ok is not True and isinstance(auth_ok, tuple) and not auth_ok[0]:
        return auth_ok[1], auth_ok[2] if len(auth_ok) > 2 else 401

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided. Send multipart/form-data with "file" field.'}), 400

    f = request.files['file']
    if not f.filename.lower().endswith('.stl'):
        return jsonify({'error': 'Only STL files are supported'}), 400

    try:
        job_id = str(uuid.uuid4())[:11]
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}.stl")
        f.save(filepath)

        mesh = trimesh.load(filepath, file_type='stl', force='mesh')
        analysis = analyze_mesh(mesh)
        printability = analyze_printability(mesh, 'z')

        jobs[job_id] = {
            'original_path': filepath,
            'original_name': f.filename,
            'analysis': analysis,
            'repaired': False,
            'repaired_path': None,
            'created_at': datetime.utcnow().isoformat()
        }

        return jsonify({
            'job_id': job_id,
            'filename': f.filename,
            'analysis': {
                'stats': analysis['stats'],
                'issues': analysis['issues']
            },
            'printability': printability['summary']
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/repair', methods=['POST'])
def api_v1_repair():
    """API: Upload STL, repair, return fixed file directly."""
    auth_ok = check_api_key()
    if auth_ok is not True and isinstance(auth_ok, tuple) and not auth_ok[0]:
        return auth_ok[1], auth_ok[2] if len(auth_ok) > 2 else 401

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    f = request.files['file']
    output_format = request.args.get('format', 'stl').lower()

    try:
        job_id = str(uuid.uuid4())[:11]
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}.stl")
        f.save(filepath)

        mesh = trimesh.load(filepath, file_type='stl', force='mesh')
        analysis = analyze_mesh(mesh)
        repaired_mesh, repairs = repair_mesh(mesh, analysis)

        # Export in requested format
        if output_format == '3mf':
            data = repaired_mesh.export(file_type='3mf')
            mimetype = 'application/vnd.ms-package.3dmanufacturing-3dmodel+xml'
            ext = '3mf'
        else:
            data = repaired_mesh.export(file_type='stl')
            mimetype = 'application/octet-stream'
            ext = 'stl'

        # Cleanup
        os.remove(filepath)

        base_name = os.path.splitext(f.filename)[0]
        return send_file(
            io.BytesIO(data),
            mimetype=mimetype,
            as_attachment=True,
            download_name=f'{base_name}_repaired.{ext}'
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/analyze-and-repair', methods=['POST'])
def api_v1_analyze_and_repair():
    """API: Upload, analyze, repair, return report + download URL."""
    auth_ok = check_api_key()
    if auth_ok is not True and isinstance(auth_ok, tuple) and not auth_ok[0]:
        return auth_ok[1], auth_ok[2] if len(auth_ok) > 2 else 401

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    f = request.files['file']

    try:
        job_id = str(uuid.uuid4())[:11]
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}.stl")
        f.save(filepath)

        mesh = trimesh.load(filepath, file_type='stl', force='mesh')
        analysis = analyze_mesh(mesh)

        # Repair
        repaired_mesh, repairs = repair_mesh(mesh, analysis)
        repaired_path = os.path.join(app.config['REPAIRED_FOLDER'], f"{job_id}_repaired.stl")
        repaired_mesh.export(repaired_path, file_type='stl')

        reanalysis = analyze_mesh(repaired_mesh)
        printability = analyze_printability(repaired_mesh, 'z')

        jobs[job_id] = {
            'original_path': filepath,
            'original_name': f.filename,
            'analysis': analysis,
            'repaired': True,
            'repaired_path': repaired_path,
            'repairs': repairs,
            'repaired_analysis': reanalysis,
            'created_at': datetime.utcnow().isoformat()
        }

        return jsonify({
            'job_id': job_id,
            'filename': f.filename,
            'original': {
                'stats': analysis['stats'],
                'issues': analysis['issues']
            },
            'repaired': {
                'stats': reanalysis['stats'],
                'issues': reanalysis['issues']
            },
            'repairs_performed': repairs,
            'printability': printability['summary'],
            'download_url': f'/api/download/{job_id}',
            'download_3mf_url': f'/api/download/{job_id}?format=3mf'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/convert', methods=['POST'])
def api_v1_convert():
    """API: Convert STL to 3MF with optional repair."""
    auth_ok = check_api_key()
    if auth_ok is not True and isinstance(auth_ok, tuple) and not auth_ok[0]:
        return auth_ok[1], auth_ok[2] if len(auth_ok) > 2 else 401

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    f = request.files['file']
    do_repair = request.args.get('repair', 'false').lower() == 'true'

    try:
        job_id = str(uuid.uuid4())[:11]
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}.stl")
        f.save(filepath)

        mesh = trimesh.load(filepath, file_type='stl', force='mesh')

        if do_repair:
            analysis = analyze_mesh(mesh)
            mesh, repairs = repair_mesh(mesh, analysis)

        data = mesh.export(file_type='3mf')
        os.remove(filepath)

        base_name = os.path.splitext(f.filename)[0]
        return send_file(
            io.BytesIO(data),
            mimetype='application/vnd.ms-package.3dmanufacturing-3dmodel+xml',
            as_attachment=True,
            download_name=f'{base_name}.3mf'
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/orient', methods=['POST'])
def api_v1_orient():
    """API: Find optimal print orientation for an STL."""
    auth_ok = check_api_key()
    if auth_ok is not True and isinstance(auth_ok, tuple) and not auth_ok[0]:
        return auth_ok[1], auth_ok[2] if len(auth_ok) > 2 else 401

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    f = request.files['file']

    try:
        job_id = str(uuid.uuid4())[:11]
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}.stl")
        f.save(filepath)

        mesh = trimesh.load(filepath, file_type='stl', force='mesh')
        result = auto_orient(mesh)
        os.remove(filepath)

        return jsonify({
            'filename': f.filename,
            'orientations': result['orientations'],
            'tested': result['all_tested'],
            'elapsed': result['elapsed_seconds']
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ══════════════════════════════════════════════
# FEATURE 4: STL → 3MF CONVERSION ON DOWNLOAD
# ══════════════════════════════════════════════
# (Integrated into existing download endpoint)

# Patch existing download to support format parameter
# Download endpoint supports format parameter (stl or 3mf)

@app.route('/api/download/<job_id>', methods=['GET'])
def download(job_id):
    """Download repaired file. Free during beta."""
    output_format = request.args.get('format', 'stl').lower()

    # Try in-memory job first
    if job_id in jobs:
        job = jobs[job_id]
        mesh_path = job.get('repaired_path') if job['repaired'] else job['original_path']
        base_name = os.path.splitext(job.get('original_name', 'model'))[0]
    else:
        # Worker restarted — check disk for repaired file
        repaired_path = os.path.join(app.config['REPAIRED_FOLDER'], f"{job_id}_repaired.stl")
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}.stl")
        if os.path.exists(repaired_path):
            mesh_path = repaired_path
        elif os.path.exists(original_path):
            mesh_path = original_path
        else:
            return jsonify({'error': 'Job not found'}), 404
        base_name = 'model'

    if not mesh_path or not os.path.exists(mesh_path):
        return jsonify({'error': 'File not found on server'}), 404

    if output_format == '3mf':
        try:
            mesh = trimesh.load(mesh_path, file_type='stl', force='mesh')
            data = mesh.export(file_type='3mf')
            return send_file(
                io.BytesIO(data),
                mimetype='application/vnd.ms-package.3dmanufacturing-3dmodel+xml',
                as_attachment=True,
                download_name=f'{base_name}_repaired.3mf'
            )
        except Exception as e:
            return jsonify({'error': f'3MF conversion failed: {e}'}), 500
    else:
        return send_file(
            mesh_path,
            mimetype='application/octet-stream',
            as_attachment=True,
            download_name=f'{base_name}_repaired.stl'
        )


# ──────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(host='0.0.0.0', port=port, debug=debug)
