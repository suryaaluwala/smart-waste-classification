"""HTML and CSS for the animated parts of the Streamlit app. Plain strings, no extra dependencies."""
from __future__ import annotations

import base64
import html
import io

from PIL import Image

INK = "#17301f"
MUTED = "#5d7265"
LINE = "#dfe9e1"
GREEN = "#1f9d55"

# label, emoji, stream, bin colour, what to do
WASTE_GUIDE = {
    "battery": ("Battery", "🔋", "Domestic hazardous", "#d64545",
                "Never put batteries in a household bin: they can leak or catch fire. Hand them in at an e-waste or battery collection point."),
    "cardboard": ("Cardboard", "📦", "Dry waste", "#2f6fde",
                  "Flatten it and keep it dry. Greasy pizza boxes go with wet waste instead."),
    "glass": ("Glass", "🍾", "Dry waste", "#2f6fde",
              "Rinse bottles and jars. Wrap broken glass in paper so that it cannot hurt waste workers."),
    "metal": ("Metal", "🥫", "Dry waste", "#2f6fde",
              "Rinse cans and tins. Clean metal is valuable to recyclers and the local kabadiwala."),
    "organic": ("Organic", "🍌", "Wet waste", "#1f9d55",
                "Food scraps and leaves belong in the wet bin. They can also be composted at home."),
    "paper": ("Paper", "📰", "Dry waste", "#2f6fde",
              "Keep it clean and dry. Tissues and food-soiled paper do not recycle well."),
    "plastic": ("Plastic", "🧴", "Dry waste", "#2f6fde",
                "Rinse and empty it first. Acceptance depends on the plastic type and your local collector."),
    "textile": ("Clothes & shoes", "👕", "Reuse or donate", "#8a5cd6",
                "If it is wearable, donate it. Worn-out fabric goes with dry waste where textile collection is not available."),
    "trash": ("General trash", "🗑️", "Reject waste", "#59636e",
              "A mixed, non-recyclable item, such as a wrapper or a sanitary product. Use the general bin."),
}


def image_uri(image: Image.Image, size: int = 320) -> str:
    copy = image.copy()
    copy.thumbnail((size, size))
    buffer = io.BytesIO()
    copy.save(buffer, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


PAGE_CSS = f"""
<style>
.block-container {{ padding-top: 3.6rem; max-width: 1040px; }}
.hero {{ border-radius: 22px; padding: 28px 30px; color: #fff; position: relative; overflow: hidden;
  background: linear-gradient(120deg, #0f7a45, #1f9d55, #15803d, #2f9e8f); background-size: 300% 300%;
  animation: drift 14s ease infinite; }}
.hero h1 {{ color: #fff; font-size: 2.05rem; margin: 0 0 4px; padding: 0; line-height: 1.2; }}
.hero p {{ margin: 0; opacity: .94; font-size: 1.04rem; max-width: 640px; }}
.hero .float {{ position: absolute; font-size: 2.1rem; opacity: .32; animation: bob 5s ease-in-out infinite; }}
.chips {{ margin-top: 14px; display: flex; flex-wrap: wrap; gap: 8px; }}
.chips span {{ background: rgba(255,255,255,.18); border-radius: 999px; padding: 3px 12px; font-size: .86rem; }}
.tip {{ border-left: 5px solid var(--c); background: color-mix(in srgb, var(--c) 9%, white); border-radius: 12px;
  padding: 14px 18px; margin: 10px 0 6px; animation: rise .6s ease both; color: {INK}; }}
.tip b {{ color: var(--c); }}
.stat {{ background: #fff; border: 1px solid {LINE}; border-radius: 16px; padding: 16px 18px; height: 100%;
  animation: rise .6s ease both; }}
.stat .v {{ font-size: 1.9rem; font-weight: 700; color: {GREEN}; line-height: 1.15; }}
.stat .l {{ color: {MUTED}; font-size: .9rem; }}
@keyframes drift {{ 0%,100% {{ background-position: 0% 50%; }} 50% {{ background-position: 100% 50%; }} }}
@keyframes bob {{ 0%,100% {{ transform: translateY(0) rotate(-6deg); }} 50% {{ transform: translateY(-12px) rotate(6deg); }} }}
@keyframes rise {{ from {{ opacity: 0; transform: translateY(10px); }} to {{ opacity: 1; transform: none; }} }}
@media (max-width: 640px) {{ .hero .float {{ display: none; }} }}
@media (prefers-reduced-motion: reduce) {{ * {{ animation: none !important; }} }}
</style>
"""


def hero(accuracy_text: str) -> str:
    return f"""<div class="hero">
  <span class="float" style="right:6%;top:14%">♻️</span>
  <span class="float" style="right:19%;top:54%;animation-delay:-2s">🥫</span>
  <span class="float" style="right:32%;top:12%;animation-delay:-3.5s">📦</span>
  <h1>Smart Waste Classifier</h1>
  <p>Show it a photo of an item. It works out what the material is and which bin it belongs in.</p>
  <div class="chips"><span>9 waste types</span><span>19,000+ photos</span><span>{html.escape(accuracy_text)}</span></div>
</div>"""


def result_animation(crop: Image.Image, ranking: list[tuple[str, float]], best: str, certain: bool) -> str:
    """Photo → network → scores → falling into the right bin, played once per prediction."""
    label, emoji, stream, colour, _ = WASTE_GUIDE[best]
    if not certain:
        colour, stream = "#c98a12", "Not sure"
    bars = ""
    for position, (name, score) in enumerate(ranking[:5]):
        bar_colour = colour if position == 0 else "#b9c9bd"
        bars += f"""<div class="row" style="--d:{2.1 + position * .14:.2f}s">
          <span class="name">{WASTE_GUIDE[name][1]} {html.escape(WASTE_GUIDE[name][0])}</span>
          <span class="track"><i style="--w:{max(score * 100, 1):.1f}%;background:{bar_colour}"></i></span>
          <span class="pct">{score:.0%}</span></div>"""
    layers = "".join(f'<b style="height:{h}%;animation-delay:{1 + i * .1:.1f}s"></b>'
                     for i, h in enumerate((100, 88, 74, 60, 46, 34, 22)))
    photo = image_uri(crop, 240)
    return f"""<!doctype html><meta charset="utf-8"><style>
*{{box-sizing:border-box}} body{{margin:0;font:15px/1.35 system-ui,'Segoe UI',sans-serif;color:{INK};background:transparent}}
.grid{{display:grid;grid-template-columns:150px 136px 1fr 150px;gap:14px;align-items:start;height:258px}}
.col h4{{font-size:.76rem;letter-spacing:.04em;text-transform:uppercase;color:{MUTED};margin:0 0 10px;font-weight:600;white-space:nowrap;
  opacity:0;animation:in .4s ease forwards;animation-delay:var(--h)}}
.photo{{width:150px;height:150px;border-radius:14px;object-fit:cover;box-shadow:0 8px 22px #17301f26;animation:scan .9s ease both .1s}}
.net{{display:flex;align-items:center;gap:6px;height:150px}} .net b{{flex:1;background:linear-gradient(#34c07a,#1f9d55);border-radius:5px;opacity:.14;
  animation:pulse 1.1s ease forwards}}
.row{{display:grid;grid-template-columns:128px 1fr 40px;gap:8px;align-items:center;margin:7px 0;opacity:0;animation:in .35s ease forwards;animation-delay:var(--d)}}
.name{{font-size:.86rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}} .pct{{font-size:.84rem;text-align:right;font-variant-numeric:tabular-nums}}
.track{{height:11px;background:#edf3ee;border-radius:99px;overflow:hidden}} .track i{{display:block;height:100%;border-radius:99px;width:0;animation:grow .7s cubic-bezier(.2,.8,.2,1) forwards;animation-delay:inherit}}
.row .track i{{animation-delay:var(--d)}}
.bin{{position:relative;width:150px;height:190px;margin:auto}} .bin svg{{position:absolute;left:25px;bottom:0;width:100px}}
.lid{{transform-origin:12px 10px;animation:lid 1.5s ease forwards 3s}}
.drop{{position:absolute;left:47px;top:-6px;width:56px;height:56px;border-radius:50%;object-fit:cover;border:3px solid #fff;box-shadow:0 4px 12px #0003;opacity:0;animation:drop 1.05s cubic-bezier(.55,0,.8,.5) forwards 3.2s}}
.stream{{text-align:center;font-weight:700;color:{colour};font-size:.92rem;opacity:0;animation:in .4s ease forwards 4.05s}}
@keyframes scan{{from{{opacity:0;transform:scale(.8)}}to{{opacity:1;transform:none}}}}
@keyframes pulse{{0%{{opacity:.14}}45%{{opacity:1;transform:scaleY(1.12)}}100%{{opacity:.62}}}}
@keyframes in{{from{{opacity:0;transform:translateY(5px)}}to{{opacity:1;transform:none}}}}
@keyframes grow{{to{{width:var(--w)}}}}
@keyframes lid{{0%{{transform:none}}30%,72%{{transform:rotate(-34deg)}}100%{{transform:none}}}}
@keyframes drop{{0%{{opacity:1;transform:none}}82%{{opacity:1;transform:translateY(84px) scale(.72)}}100%{{opacity:0;transform:translateY(112px) scale(.45)}}}}
@media (max-width:640px){{.grid{{grid-template-columns:1fr 1fr;height:auto}}.bars{{grid-column:1/-1}}.net{{display:none}}.netcol{{display:none}}.photo{{width:120px;height:120px}}.bin{{height:170px}}}}
@media (prefers-reduced-motion:reduce){{*{{animation-duration:.01s!important;animation-delay:0s!important}}.track i{{width:var(--w)}}}}
</style>
<div class="grid">
  <div class="col"><h4 style="--h:0s">1. Your photo</h4><img class="photo" src="{photo}" alt="The photo being classified"></div>
  <div class="col netcol"><h4 style="--h:.9s">2. Neural network</h4><div class="net">{layers}</div></div>
  <div class="col bars"><h4 style="--h:1.9s">3. Its scores</h4>{bars}</div>
  <div class="col"><h4 style="--h:2.9s;text-align:center">4. Sorted</h4><div class="bin">
    <img class="drop" src="{photo}" alt="">
    <svg viewBox="0 0 100 124" aria-hidden="true">
      <g class="lid"><rect x="4" y="8" width="92" height="10" rx="4" fill="{colour}"/><rect x="38" y="1" width="24" height="8" rx="3" fill="{colour}"/></g>
      <path d="M12 22h76l-7 96a6 6 0 0 1-6 6H25a6 6 0 0 1-6-6z" fill="{colour}"/>
      <path d="M35 40l3 66M50 40v66M65 40l-3 66" stroke="#fff" stroke-opacity=".38" stroke-width="4" stroke-linecap="round"/>
    </svg></div><div class="stream">{emoji if certain else "❓"} {html.escape(stream)}</div></div>
</div>"""


HOW_IT_WORKS = f"""<!doctype html><meta charset="utf-8"><style>
*{{box-sizing:border-box}} body{{margin:0;font:15px/1.45 system-ui,'Segoe UI',sans-serif;color:{INK};background:transparent}}
.steps{{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:12px}}
button{{font:inherit;font-size:.88rem;border:1px solid {LINE};background:#fff;color:{INK};padding:7px 12px;border-radius:999px;cursor:pointer;transition:.2s}}
button.on{{background:{GREEN};border-color:{GREEN};color:#fff}}
.stage{{height:208px;border:1px solid {LINE};background:#fff;border-radius:18px;position:relative;overflow:hidden}}
.say{{min-height:74px;padding:12px 4px 0}} .say b{{display:block;font-size:1.06rem;margin-bottom:2px}}
.grid{{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;gap:12px}}
.pix{{display:grid;grid-template-columns:repeat(8,17px);grid-auto-rows:17px;gap:2px}} .pix i{{border-radius:3px;animation:pop .5s ease both}}
.layer{{width:34px;border-radius:7px;background:linear-gradient(#34c07a,#178a49);animation:flow 1.8s ease-in-out infinite}}
.tag{{font-size:.8rem;color:{MUTED};text-align:center;margin-top:7px}} .stack{{display:flex;flex-direction:column;align-items:center}}
.chip{{background:#eef6f0;border-radius:10px;padding:7px 12px;font-size:.92rem;animation:pop .45s ease both}}
.bar{{display:grid;grid-template-columns:96px 300px 42px;gap:8px;align-items:center;font-size:.86rem;margin:5px 0}} .bar s{{display:block;height:11px;border-radius:9px;background:#edf3ee;overflow:hidden}}
.bar s i{{display:block;height:100%;border-radius:9px;background:{GREEN};width:0;animation:gw .9s ease forwards}}
.arrow{{font-size:1.5rem;color:{GREEN};animation:nudge 1.1s ease-in-out infinite}}
@keyframes pop{{from{{opacity:0;transform:scale(.6)}}to{{opacity:1;transform:none}}}}
@keyframes flow{{0%,100%{{opacity:.5}}50%{{opacity:1;transform:scaleY(1.07)}}}}
@keyframes gw{{to{{width:var(--w)}}}} @keyframes nudge{{50%{{transform:translateX(5px)}}}}
@media (max-width:640px){{.bar{{grid-template-columns:82px 1fr 38px}}.chip{{font-size:.8rem;padding:5px 8px}}.layer{{width:22px}}.grid{{gap:7px}}}}
@media (prefers-reduced-motion:reduce){{*{{animation-duration:.01s!important;animation-iteration-count:1!important}}.bar s i{{width:var(--w)}}}}
</style>
<div class="steps" id="steps"></div><div class="stage"><div class="grid" id="stage"></div></div><div class="say" id="say"></div>
<script>
const colours=['#c8a26b','#b78d55','#a57a44','#d9bb8b','#8f6a3a','#e5cda4','#7c5a30','#c49c63'];
const pixels=()=>'<div class="pix">'+Array.from({{length:40}},(_,i)=>`<i style="background:${{colours[(i*7+(i%5)*3)%8]}};animation-delay:${{i*12}}ms"></i>`).join('')+'</div>';
const layers=[120,104,88,72,56,42,28].map((h,i)=>`<div class="layer" style="height:${{h}}px;animation-delay:${{i*.13}}s"></div>`).join('');
const chips=(a)=>a.map((t,i)=>`<div class="chip" style="animation-delay:${{i*.13}}s">${{t}}</div>`).join('<span class="arrow">→</span>');
const bars=[['Cardboard',86],['Paper',8],['Plastic',3],['Trash',2]].map(([n,v],i)=>`<div class="bar"><span>${{n}}</span><s><i style="--w:${{v}}%;animation-delay:${{i*.15}}s"></i></s><span>${{v}}%</span></div>`).join('');
const S=[
 ['Photo','A photo is just numbers',"A computer sees a grid of tiny coloured squares called pixels. Each one is three numbers: red, green and blue.",()=>pixels()+'<span class="arrow">→</span><div class="stack"><div class="chip">R 200 · G 162 · B 107</div><div class="tag">one pixel</div></div>'],
 ['Layers','The network spots patterns',"The photo passes through more than a dozen stacked blocks. Early ones find edges, middle ones find textures, and later ones find whole shapes.",()=>`<div class="stack"><div style="display:flex;align-items:center;gap:7px">${{layers}}</div><div class="tag">MobileNetV3: small enough to run on a laptop or phone</div></div>`],
 ['Learning','It learned from examples',"The model studied about 13,500 labelled photos. Each time it guessed wrong, it adjusted its internal numbers slightly. This is called training.",()=>chips(['Guess: paper','Actual: cardboard','Adjust','Try again'])],
 ['Scores','It scores every class',"The last layer gives one score per waste type, and the scores add up to 100%. The highest score is the prediction. Below the safety line, the app says that it is not sure.",()=>`<div>${{bars}}</div>`],
 ['Explain','It shows where it looked',"Grad-CAM highlights the parts of the photo that drove the decision. It lets you check that the model looked at the object and not at the background.",()=>chips(['Prediction','Trace it back','Heatmap','Can we trust it?'])],
 ['Bin','The result becomes an action',"The material maps to a waste stream: wet, dry, hazardous, reuse or reject. Separating waste correctly at home is what makes recycling possible.",()=>chips(['Banana peel → Wet','Tin can → Dry','Battery → Hazardous'])],
];
let at=0,timer;
const nav=document.getElementById('steps');
S.forEach((s,i)=>{{const b=document.createElement('button');b.textContent=(i+1)+'. '+s[0];b.onclick=()=>{{clearInterval(timer);show(i)}};nav.append(b)}});
function show(i){{at=i;[...nav.children].forEach((b,j)=>b.classList.toggle('on',j===i));document.getElementById('stage').innerHTML=S[i][3]();document.getElementById('say').innerHTML=`<b>${{S[i][1]}}</b>${{S[i][2]}}`}}
show(0);timer=setInterval(()=>show((at+1)%S.length),6500);
</script>"""
