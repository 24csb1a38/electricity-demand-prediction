// =========================================
// GridPulse JavaScript
// =========================================

// Smooth Scroll
document.querySelectorAll('a[href^="#"]').forEach(anchor => {

    anchor.addEventListener("click", function(e){

        e.preventDefault();

        document.querySelector(this.getAttribute("href"))
        .scrollIntoView({
            behavior:"smooth"
        });

    });

});

// =========================================
// Scroll To Top
// =========================================

const scrollBtn = document.getElementById("scrollTop");

window.addEventListener("scroll",()=>{

    if(window.scrollY>500){

        scrollBtn.style.display="block";

    }

    else{

        scrollBtn.style.display="none";

    }

});

scrollBtn.onclick=()=>{

    window.scrollTo({

        top:0,

        behavior:"smooth"

    });

};

// =========================================
// Animated Counters
// =========================================

function animateCounter(id,end,suffix=""){

    let element=document.getElementById(id);

    if(!element) return;

    let start=0;

    let duration=2000;

    let increment=end/(duration/20);

    let timer=setInterval(()=>{

        start+=increment;

        if(start>=end){

            start=end;

            clearInterval(timer);

        }

        element.innerHTML=Math.floor(start)+suffix;

    },20);

}

animateCounter("days",724);

animateCounter("accuracy",94,"%");

animateCounter("models",3);

animateCounter("features",7);

// =========================================
// Fade In Animation
// =========================================

const observer=new IntersectionObserver(entries=>{

    entries.forEach(entry=>{

        if(entry.isIntersecting){

            entry.target.classList.add("show");

        }

    });

});

document.querySelectorAll("section").forEach(sec=>{

    sec.classList.add("hidden");

    observer.observe(sec);

});

// =========================================
// Dashboard Stats
// =========================================

fetch("/api/stats")

.then(response=>response.json())

.then(data=>{

    if(data.error) return;

    document.getElementById("maxDemand").innerHTML=data.max_demand+" MW";

    document.getElementById("avgDemand").innerHTML=data.average_demand+" MW";

    document.getElementById("totalDays").innerHTML=data.rows;

});

// =========================================
// Model Comparison
// =========================================

fetch("/api/models")

.then(response=>response.json())

.then(models=>{

    let table=document.getElementById("modelTable");

    if(!table) return;

    table.innerHTML="";

    models.forEach(model=>{

        let badge="";

        if(model.model==="XGBoost")

            badge="Best";

        else if(model.model==="Prophet")

            badge="Good";

        else

            badge="Baseline";

        table.innerHTML+=`

        <tr>

            <td>${model.model}</td>

            <td>${model.MAE}</td>

            <td>${model.RMSE}</td>

            <td>${badge}</td>

        </tr>

        `;

    });

});

// =========================================
// Plotly Demand Chart
// =========================================

fetch("/api/chart")

.then(response=>response.json())

.then(data=>{

    let x=[];

    let y=[];

    data.forEach(row=>{

        x.push(row.date);

        y.push(row.demand);

    });

    let trace={

        x:x,

        y:y,

        mode:"lines",

        line:{

            color:"#4F8CFF",

            width:3

        },

        fill:"tozeroy",

        fillcolor:"rgba(79,140,255,0.15)"

    };

    let layout={

        title:"Telangana Electricity Demand",

        paper_bgcolor:"rgba(0,0,0,0)",

        plot_bgcolor:"rgba(0,0,0,0)",

        font:{

            color:"#F8FAFC"

        },

        xaxis:{

            showgrid:false

        },

        yaxis:{

            gridcolor:"rgba(255,255,255,.08)"

        },

        margin:{

            l:40,

            r:20,

            t:40,

            b:40

        }

    };

    Plotly.newPlot(

        "demandChart",

        [trace],

        layout,

        {

            responsive:true,

            displayModeBar:false

        }

    );

});

// =========================================
// Navbar Highlight
// =========================================

const sections=document.querySelectorAll("section");

const navLinks=document.querySelectorAll("nav ul li a");

window.addEventListener("scroll",()=>{

    let current="";

    sections.forEach(sec=>{

        const top=sec.offsetTop-150;

        if(pageYOffset>=top){

            current=sec.getAttribute("id");

        }

    });

    navLinks.forEach(link=>{

        link.classList.remove("active");

        if(link.getAttribute("href")==="#"+current){

            link.classList.add("active");

        }

    });

});

// =========================================
// Floating Cards
// =========================================

document.querySelectorAll(".dashboard-card").forEach(card=>{

    card.addEventListener("mousemove",e=>{

        let x=e.offsetX/card.offsetWidth-.5;

        let y=e.offsetY/card.offsetHeight-.5;

        card.style.transform=

        `rotateY(${x*8}deg) rotateX(${-y*8}deg)`;

    });

    card.addEventListener("mouseleave",()=>{

        card.style.transform="rotateY(0deg) rotateX(0deg)";

    });

});

console.log("⚡ GridPulse Loaded Successfully");