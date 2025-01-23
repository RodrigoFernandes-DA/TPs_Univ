<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
    <xsl:output method="html" indent="yes" />
    <xsl:template match="/">
        <html>
            <head>
                <link href="style.css" rel="stylesheet" type="text/css"/>
                <title>Le Seigneur des Anneaux</title>
            </head>

            <body>
                <h1>Le Seigneur des Anneaux</h1>
                <div class="film-div">
                    <xsl:apply-templates select="lord-of-the-rings/films/film" />
                </div>
            </body>
        </html>
    </xsl:template>

    <xsl:template match="film">
        <div class="film-title-div">
            <xsl:value-of select="title" />
        </div>
        <div class="film-content-div">
            <div class="film-left-div">
                <img src="{img-source/@value}" alt="{nombre}" width="180" height="270"/>
            </div>
            <div class="film-right-div">
                <p> <b>Rélasiteur</b> : <xsl:value-of select="director"/> </p>
                <p> <b>Date de sortie</b> : <xsl:value-of select="release-date"/> </p>
                <p> <b>Box-office</b>: 
                    <xsl:value-of select="format-number(box-office div 1000000, '#.##')"/> millions USD (
                    <span class="stars">
                        <xsl:call-template name="box-office-stars">
                            <xsl:with-param name="gain" select="box-office div 1000000"/>
                        </xsl:call-template>
                    </span> )</p>
                <p> <b>Roles principaux</b> :  </p>
                <ul>
                    <xsl:apply-templates select="roles/role" />
                </ul>
            </div>  
        </div>
    </xsl:template>
    
    <xsl:template match="roles/role">
        <li>
            <xsl:value-of select="@character"/> - 
            <xsl:apply-templates select="/lord-of-the-rings/actors/actor[@id=current()/@actor]" />
        </li>
    </xsl:template>

    <xsl:template match="actors/actor">
            <a href="{wiki-page}">
                <xsl:value-of select="name"/>
            </a>
    </xsl:template>

    <xsl:template name="box-office-stars">
        <xsl:param name="gain"/>
        <xsl:choose>
            <xsl:when test="$gain > 1000">
                <xsl:text>★★★★★</xsl:text>
            </xsl:when>
            <xsl:when test="$gain > 500">
                <xsl:text>★★★★</xsl:text>
            </xsl:when>
            <xsl:when test="$gain > 100">
                <xsl:text>★★★</xsl:text>
            </xsl:when>
            <xsl:when test="$gain > 50">
                <xsl:text>★★</xsl:text>
            </xsl:when>
            <xsl:otherwise>
                <xsl:text>★</xsl:text>
            </xsl:otherwise>
        </xsl:choose>
    </xsl:template>


</xsl:stylesheet>