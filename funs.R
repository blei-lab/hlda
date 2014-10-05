plot.scores <- function(filename, ...)
{
    x <- read.table(filename);
    par(mfrow=c(2,2));
    plot.score(x[,2], title="GEM");
    plot.score(x[,3], title="ETA");
    plot.score(x[,4], title="GAMMA");
    plot.score(x[,5], title="TOTAL");
}


plot.score <- function(v, title)
{
    plot(v, type="b", col="blue", lwd=2, main=title)
    abline(h=max(v), col="gray", lwd=2, lty=2);
    points(which.max(v), max(v), col="red", pch=16.0, cex=2);
}


monitor.scores <- function(filename, lag = 15)
{
    while (TRUE)
    {
        cat("plotting\n");
        plot.scores(filename, lwd=2, col="red");
        Sys.sleep(lag);
    }
}


compute.gamma <- function(n, k.1, k.2)
{
    gam.1 <- k.1 / log(n);
    gam.2 <- k.2 / (log(n) - log(gam.1) - log(log(n)))
    return(c(gam.1, gam.2))
}
