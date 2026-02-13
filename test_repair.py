"""
SliceReady Repair Engine — Pressure Test Suite
Tests every failure mode we've seen in real AI-generated STLs.
"""
import sys
import time
import traceback
import numpy as np
import trimesh

# We'll import the repair engine after building it
# For now, build test meshes

def make_clean_box():
    """Perfect watertight box — should pass through unchanged."""
    return trimesh.creation.box(extents=[20, 15, 10])

def make_holey_box():
    """Box with faces removed — needs hole filling."""
    box = trimesh.creation.box(extents=[20, 15, 10])
    keep = np.ones(len(box.faces), dtype=bool)
    keep[0:4] = False  # Remove 4 faces
    box.update_faces(keep)
    return box

def make_multi_body():
    """Multiple disconnected components — all significant, should keep all."""
    parts = []
    for i in range(5):
        b = trimesh.creation.box(extents=[8, 8, 8])
        b.apply_translation([i * 15, 0, 0])
        parts.append(b)
    return trimesh.util.concatenate(parts)

def make_multi_body_with_debris():
    """Large main body + tiny floating debris — should remove only debris."""
    main = trimesh.creation.box(extents=[50, 50, 50])
    debris1 = trimesh.creation.box(extents=[0.5, 0.5, 0.5])
    debris1.apply_translation([100, 0, 0])
    debris2 = trimesh.creation.icosphere(radius=0.3)
    debris2.apply_translation([0, 100, 0])
    return trimesh.util.concatenate([main, debris1, debris2])

def make_inverted_normals():
    """Mesh with flipped normals."""
    box = trimesh.creation.box(extents=[20, 15, 10])
    box.invert()
    return box

def make_degenerate_faces():
    """Mesh with zero-area triangles mixed in."""
    box = trimesh.creation.box(extents=[20, 15, 10])
    # Add degenerate faces (3 vertices at same point)
    n = len(box.vertices)
    extra_verts = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0],
                            [1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=float)
    extra_faces = np.array([[n, n+1, n+2], [n+3, n+4, n+5]])
    new_verts = np.vstack([box.vertices, extra_verts])
    new_faces = np.vstack([box.faces, extra_faces])
    return trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=False)

def make_duplicate_faces():
    """Mesh with duplicate triangles."""
    box = trimesh.creation.box(extents=[20, 15, 10])
    # Duplicate first 6 faces
    new_faces = np.vstack([box.faces, box.faces[:6]])
    return trimesh.Trimesh(vertices=box.vertices, faces=new_faces, process=False)

def make_high_poly_sphere():
    """High-poly sphere (like AI output) — should NOT auto-decimate."""
    return trimesh.creation.icosphere(subdivisions=5)  # ~20K faces

def make_very_high_poly():
    """Very high poly mesh — should still NOT decimate during repair."""
    s = trimesh.creation.icosphere(subdivisions=6)  # ~80K faces
    return s

def make_non_manifold():
    """Mesh with non-manifold edges (edges shared by >2 faces)."""
    box = trimesh.creation.box(extents=[20, 15, 10])
    # Add extra face sharing an existing edge
    v = box.vertices
    n = len(v)
    # Add a point sticking out
    extra_vert = np.array([[0, 0, 20]])
    new_verts = np.vstack([v, extra_vert])
    # Create a face using two vertices from box + the new one
    extra_face = np.array([[0, 1, n]])
    new_faces = np.vstack([box.faces, extra_face])
    return trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=False)

def make_complex_ai_model():
    """Simulate AI model: multiple overlapping shapes, non-manifold, high poly."""
    parts = []
    # Main body
    body = trimesh.creation.cylinder(radius=10, height=30, sections=64)
    parts.append(body)
    # Overlapping sphere (creates non-manifold intersection)
    head = trimesh.creation.icosphere(radius=8, subdivisions=3)
    head.apply_translation([0, 0, 15])
    parts.append(head)
    # Arms
    for x in [-1, 1]:
        arm = trimesh.creation.cylinder(radius=3, height=20, sections=32)
        arm.apply_translation([x * 13, 0, 5])
        parts.append(arm)
    # Small decorative elements (should be kept!)
    for i in range(8):
        angle = i * np.pi / 4
        dec = trimesh.creation.icosphere(radius=2, subdivisions=2)
        dec.apply_translation([12 * np.cos(angle), 12 * np.sin(angle), 0])
        parts.append(dec)
    return trimesh.util.concatenate(parts)

def make_thin_walls():
    """Mesh simulating thin walls common in AI models."""
    # Create a thin box (0.2mm wall in one direction)
    return trimesh.creation.box(extents=[20, 15, 0.2])

def make_open_surface():
    """A completely open surface (like a terrain or relief) — no holes to fill, just an open mesh."""
    # Create a flat grid
    x = np.linspace(0, 20, 30)
    y = np.linspace(0, 15, 25)
    xx, yy = np.meshgrid(x, y)
    zz = np.sin(xx * 0.5) * np.cos(yy * 0.5) * 3
    
    verts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    faces = []
    rows, cols = 25, 30
    for i in range(rows - 1):
        for j in range(cols - 1):
            v0 = i * cols + j
            v1 = v0 + 1
            v2 = v0 + cols
            v3 = v2 + 1
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])
    return trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=False)


# ═══ STRESS TEST MESH GENERATORS ═══

def make_200k_sphere():
    """200K+ face sphere — tests memory handling without OOM."""
    # icosphere subdivisions=7 gives ~327K faces, use 6 for ~80K + duplicate
    s1 = trimesh.creation.icosphere(subdivisions=6)
    s2 = trimesh.creation.icosphere(subdivisions=6)
    s2.apply_translation([30, 0, 0])
    return trimesh.util.concatenate([s1, s2])  # ~164K faces

def make_many_overlapping_parts():
    """15 overlapping geometric parts — like a complex AI-generated figurine."""
    parts = []
    # Torso
    parts.append(trimesh.creation.cylinder(radius=8, height=25, sections=48))
    # Head
    h = trimesh.creation.icosphere(radius=6, subdivisions=3)
    h.apply_translation([0, 0, 18])
    parts.append(h)
    # Arms and legs (overlapping with torso)
    for sign in [-1, 1]:
        arm = trimesh.creation.cylinder(radius=2.5, height=18, sections=24)
        arm.apply_translation([sign * 10, 0, 8])
        parts.append(arm)
        leg = trimesh.creation.cylinder(radius=3, height=20, sections=24)
        leg.apply_translation([sign * 5, 0, -18])
        parts.append(leg)
    # Accessories: helmet, shield, sword
    helmet = trimesh.creation.icosphere(radius=7, subdivisions=2)
    helmet.apply_translation([0, 0, 20])
    parts.append(helmet)
    shield = trimesh.creation.cylinder(radius=5, height=1, sections=32)
    shield.apply_translation([-12, 0, 5])
    parts.append(shield)
    sword = trimesh.creation.cylinder(radius=0.5, height=25, sections=12)
    sword.apply_translation([12, 0, 10])
    parts.append(sword)
    # Small decorative elements (buttons, buckles)
    for i in range(6):
        btn = trimesh.creation.icosphere(radius=1, subdivisions=1)
        btn.apply_translation([0, 8, -5 + i * 4])
        parts.append(btn)
    return trimesh.util.concatenate(parts)

def make_good_with_scattered_debris():
    """Large clean model with tiny floating specks scattered around."""
    main = trimesh.creation.icosphere(radius=20, subdivisions=4)  # ~5K faces
    parts = [main]
    # Add 20 tiny specks of debris
    for i in range(20):
        speck = trimesh.creation.icosphere(radius=0.1, subdivisions=0)
        speck.apply_translation([
            np.random.uniform(-50, 50),
            np.random.uniform(-50, 50),
            np.random.uniform(-50, 50)
        ])
        parts.append(speck)
    return trimesh.util.concatenate(parts)

def make_complex_non_watertight():
    """Complex model that's non-watertight — holes punched in multiple places."""
    mesh = trimesh.creation.icosphere(radius=15, subdivisions=4)
    # Remove random faces to create holes
    n = len(mesh.faces)
    keep = np.ones(n, dtype=bool)
    # Remove ~5% of faces in scattered patches
    remove_indices = np.random.choice(n, n // 20, replace=False)
    keep[remove_indices] = False
    mesh.update_faces(keep)
    return mesh


# ═══════════════════════════════════════════════
# Test runner
# ═══════════════════════════════════════════════

TEST_CASES = [
    ("Clean box (no repair needed)", make_clean_box, {
        'should_preserve_faces': True,   # Don't add/remove significant faces
        'max_face_loss_pct': 5,
        'should_not_decimate': True,
    }),
    ("Box with holes", make_holey_box, {
        'should_fix_watertight': True,
        'max_face_loss_pct': 10,
    }),
    ("Multi-body (all significant)", make_multi_body, {
        'min_face_ratio': 0.90,  # Should keep >90% of faces
        'should_not_decimate': True,
    }),
    ("Multi-body with tiny debris", make_multi_body_with_debris, {
        'min_face_ratio': 0.85,  # OK to remove debris
        'should_not_decimate': True,
    }),
    ("Inverted normals", make_inverted_normals, {
        'max_face_loss_pct': 5,
        'should_not_decimate': True,
    }),
    ("Degenerate faces", make_degenerate_faces, {
        'max_face_loss_pct': 20,  # Degenerate faces SHOULD be removed
    }),
    ("Duplicate faces", make_duplicate_faces, {
        'max_face_loss_pct': 40,  # Will remove dupes
    }),
    ("High-poly sphere (20K)", make_high_poly_sphere, {
        'should_not_decimate': True,
        'max_face_loss_pct': 5,
    }),
    ("Very high-poly (80K)", make_very_high_poly, {
        'should_not_decimate': True,
        'max_face_loss_pct': 5,
    }),
    ("Non-manifold edges", make_non_manifold, {
        'max_face_loss_pct': 15,
        'skip_bbox_check': True,  # Non-manifold spikes get removed — bbox shrink is correct
    }),
    ("Complex AI model (multi-part)", make_complex_ai_model, {
        'min_face_ratio': 0.80,
        'should_not_decimate': True,
    }),
    ("Thin walls", make_thin_walls, {
        'max_face_loss_pct': 10,
        'should_not_decimate': True,
    }),
    ("Open surface", make_open_surface, {
        'max_face_loss_pct': 10,
    }),
    # ═══ STRESS TESTS — simulate real AI model failures ═══
    ("STRESS: 200K face sphere (memory)", make_200k_sphere, {
        'should_not_decimate': True,
        'max_face_loss_pct': 5,
    }),
    ("STRESS: 15 overlapping parts", make_many_overlapping_parts, {
        'min_face_ratio': 0.85,
        'should_not_decimate': True,
    }),
    ("STRESS: Mostly good + tiny debris", make_good_with_scattered_debris, {
        'min_face_ratio': 0.95,
        'should_not_decimate': True,
    }),
    ("STRESS: Non-watertight complex", make_complex_non_watertight, {
        'min_face_ratio': 0.80,
    }),
    ("SPEED: Clean 80K (should fast-path)", make_very_high_poly, {
        'should_not_decimate': True,
        'max_time_seconds': 1.0,
    }),
]


def analyze_for_test(mesh):
    """Minimal analysis dict for repair engine."""
    areas = mesh.area_faces
    face_sorted = np.sort(mesh.faces, axis=1)
    _, face_unique_counts = np.unique(face_sorted, axis=0, return_counts=True)
    
    edges_sorted = np.sort(mesh.edges_sorted, axis=1)
    edge_keys = edges_sorted[:, 0] * (len(mesh.vertices) + 1) + edges_sorted[:, 1]
    _, edge_counts = np.unique(edge_keys, return_counts=True)
    
    bodies = mesh.split(only_watertight=False)
    
    return {
        'stats': {
            'triangles': len(mesh.faces),
            'vertices': len(mesh.vertices),
            'is_watertight': bool(mesh.is_watertight),
            'degenerate_faces': int(np.sum(areas < 1e-10)),
            'duplicate_faces': int(np.sum(face_unique_counts > 1)),
            'non_manifold_edges': int(np.sum(edge_counts != 2)),
            'body_count': len(bodies),
            'is_high_poly': len(mesh.faces) > 100000,
        },
        'issues': []  # Not needed for repair
    }


def run_tests(repair_fn):
    """Run all test cases against a repair function."""
    results = []
    total_pass = 0
    total_fail = 0
    
    print("=" * 70)
    print("SLICEREADY REPAIR ENGINE — PRESSURE TEST SUITE")
    print("=" * 70)
    
    for name, mesh_fn, criteria in TEST_CASES:
        print(f"\n{'─' * 60}")
        print(f"TEST: {name}")
        print(f"{'─' * 60}")
        
        try:
            mesh = mesh_fn()
            original_faces = len(mesh.faces)
            original_verts = len(mesh.vertices)
            original_bounds = mesh.bounds.copy()
            original_watertight = mesh.is_watertight
            
            print(f"  Input:  {original_faces:,} faces, {original_verts:,} verts, watertight={original_watertight}")
            
            analysis = analyze_for_test(mesh)
            
            t0 = time.time()
            repaired, repairs = repair_fn(mesh.copy(), analysis)
            duration = time.time() - t0
            
            repaired_faces = len(repaired.faces)
            repaired_verts = len(repaired.vertices)
            face_ratio = repaired_faces / original_faces if original_faces > 0 else 1
            
            print(f"  Output: {repaired_faces:,} faces, {repaired_verts:,} verts, watertight={repaired.is_watertight}")
            print(f"  Ratio:  {face_ratio:.2%} of original faces preserved")
            print(f"  Time:   {duration:.2f}s")
            print(f"  Steps:  {repairs}")
            
            # Check criteria
            failures = []
            
            if 'max_face_loss_pct' in criteria:
                max_loss = criteria['max_face_loss_pct']
                actual_loss = (1 - face_ratio) * 100
                if actual_loss > max_loss:
                    failures.append(f"Face loss {actual_loss:.1f}% > max {max_loss}%")
            
            if 'min_face_ratio' in criteria:
                if face_ratio < criteria['min_face_ratio']:
                    failures.append(f"Face ratio {face_ratio:.2%} < min {criteria['min_face_ratio']:.0%}")
            
            if 'should_not_decimate' in criteria and criteria['should_not_decimate']:
                if face_ratio < 0.90:
                    failures.append(f"Unexpected decimation: {face_ratio:.2%} faces remaining")
            
            if 'should_fix_watertight' in criteria and criteria['should_fix_watertight']:
                if not repaired.is_watertight:
                    failures.append("Should be watertight but isn't")
            
            if 'should_preserve_faces' in criteria and criteria['should_preserve_faces']:
                diff = abs(repaired_faces - original_faces)
                if diff > original_faces * 0.05:
                    failures.append(f"Face count changed by {diff} (>5% of {original_faces})")
            
            # Bounding box check — repair should never shrink model significantly
            if repaired_faces > 0 and not criteria.get('skip_bbox_check', False):
                new_bounds = repaired.bounds
                orig_size = original_bounds[1] - original_bounds[0]
                new_size = new_bounds[1] - new_bounds[0]
                size_ratio = np.min(new_size / (orig_size + 1e-10))
                if size_ratio < 0.7:
                    failures.append(f"Bounding box shrunk to {size_ratio:.0%}")

            # Speed check
            if 'max_time_seconds' in criteria:
                if duration > criteria['max_time_seconds']:
                    failures.append(f"Too slow: {duration:.2f}s > {criteria['max_time_seconds']}s")
            
            if failures:
                print(f"  ❌ FAIL: {'; '.join(failures)}")
                total_fail += 1
                results.append((name, 'FAIL', failures))
            else:
                print(f"  ✅ PASS")
                total_pass += 1
                results.append((name, 'PASS', []))
                
        except Exception as e:
            print(f"  💥 CRASH: {e}")
            traceback.print_exc()
            total_fail += 1
            results.append((name, 'CRASH', [str(e)]))
    
    print(f"\n{'=' * 70}")
    print(f"RESULTS: {total_pass} passed, {total_fail} failed out of {len(TEST_CASES)}")
    print(f"{'=' * 70}")
    
    return results, total_pass, total_fail


if __name__ == '__main__':
    # Import the repair function
    sys.path.insert(0, '/home/claude/meshready')
    from app import repair_mesh, analyze_mesh
    
    results, passed, failed = run_tests(repair_mesh)
    sys.exit(0 if failed == 0 else 1)
